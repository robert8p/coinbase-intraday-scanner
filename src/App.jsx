import { useState, useEffect, useCallback, useMemo } from "react";

const SCAN_HOURS = [0,4,8,12,16,20];
const F = "'JetBrains Mono','SF Mono','Fira Code','Cascadia Code',monospace";
const Box = ({children,style})=><div style={{background:"rgba(255,255,255,0.015)",borderRadius:8,border:"1px solid rgba(255,255,255,0.04)",padding:16,...style}}>{children}</div>;
const Lbl = ({children})=><div style={{fontSize:11,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:10}}>{children}</div>;
const Btn = ({children,active,color="#3b82f6",onClick,disabled,style:s})=>(
  <button onClick={onClick} disabled={disabled} style={{padding:"4px 10px",borderRadius:4,fontSize:11,fontFamily:F,
    cursor:disabled?"default":"pointer",border:`1px solid ${active?color:"rgba(255,255,255,0.06)"}`,
    background:active?`${color}18`:"transparent",color:active?color:disabled?"#334155":"#64748b",
    opacity:disabled?0.5:1,transition:"all 0.15s",...s}}>{children}</button>);

// Format UTC hour labels: 00:00, 04:00, etc.
const fmtHour = h => String(h).padStart(2,"0") + ":00";

function SourceBadge({source,trained}) {
  const cfg={live:{c:"#22c55e",l:"● LIVE"},cached:{c:"#eab308",l:"LAST SCAN"},offline:{c:"#64748b",l:"OFFLINE"},loading:{c:"#64748b",l:"..."},error:{c:"#ef4444",l:"ERROR"}}[source]||{c:"#64748b",l:"?"};
  return (
    <div style={{display:"flex",gap:6,alignItems:"center"}}>
      <span style={{fontSize:9,padding:"2px 8px",borderRadius:3,fontWeight:700,letterSpacing:0.5,background:`${cfg.c}18`,color:cfg.c,border:`1px solid ${cfg.c}30`}}>{cfg.l}</span>
      {trained&&<span style={{fontSize:9,padding:"2px 8px",borderRadius:3,fontWeight:700,letterSpacing:0.5,background:"rgba(139,92,246,0.12)",color:"#8b5cf6",border:"1px solid rgba(139,92,246,0.2)"}}>LGBM FIRST-PASSAGE</span>}
    </div>);
}

function WinBar({winProb,ev,breakeven=0.6}) {
  const pct=(winProb*100).toFixed(1);
  const c=winProb>breakeven+0.09?"#22c55e":winProb>breakeven+0.04?"#a3e635":winProb>breakeven?"#eab308":winProb>breakeven-0.06?"#f97316":"#6b7280";
  const evColor = ev>0?"#22c55e":ev<0?"#ef4444":"#64748b";
  return (
    <div style={{display:"flex",alignItems:"center",gap:6,minWidth:180}}>
      <div style={{width:50,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden",flexShrink:0}}>
        <div style={{width:`${Math.max(2,winProb*100)}%`,height:"100%",background:c,borderRadius:3,transition:"width 0.4s"}}/>
      </div>
      <span style={{fontVariantNumeric:"tabular-nums",fontSize:12,color:c,fontWeight:600,minWidth:40}}>{pct}%</span>
      <span style={{fontVariantNumeric:"tabular-nums",fontSize:11,color:evColor,fontWeight:500,minWidth:44}}>
        {ev>0?"+":""}{ev.toFixed(2)}%
      </span>
    </div>);
}

function Fc({value,label}) {
  const v=parseFloat(value);let c="#94a3b8";
  if(["momentum","vwapDist","vwapSlope","trendStr","orbStrength"].includes(label)) c=v>0.4?"#22c55e":v>0.15?"#a3e635":v>0?"#94a3b8":v>-0.2?"#f97316":"#ef4444";
  else if(label==="relVolume") c=v>1.8?"#22c55e":v>1.2?"#a3e635":"#94a3b8";
  else if(label==="atrReach") c=v<0.8?"#22c55e":v<1.2?"#eab308":"#ef4444";
  return <span style={{color:c,fontVariantNumeric:"tabular-nums",fontSize:11.5}}>{value}</span>;
}

// ─── SCANNER ─────────────────────────────────────────────────────
function ScannerTab({data,scanHour,source,elapsed,message,modelWR10,modelPnL10,health,scanInfo}) {
  const [mode,setMode]=useState("posEV");

  if(source==="offline"||!data||data.length===0) return (
    <Box style={{padding:40,textAlign:"center"}}>
      <div style={{fontSize:14,color:"#64748b",marginBottom:8}}>{source==="offline"?"No model trained":"No data"}</div>
      <div style={{fontSize:12,color:"#475569"}}>{message||"Train model first, then scan."}</div>
    </Box>);

  const activeTP = scanInfo?.tp_pct ?? health?.tp_pct ?? 2.0;
  const activeSL = scanInfo?.sl_pct ?? health?.sl_pct ?? 1.0;
  const activeBE = scanInfo?.breakeven ?? health?.breakeven ?? (activeSL/(activeSL+activeTP)*100).toFixed(1);
  const horizon = scanInfo?.horizonHours ?? health?.horizonHours ?? 4;
  const beThresh = activeSL/(activeSL+activeTP);
  const beThresh5 = beThresh + 0.05;

  const filtered = mode==="be" ? data.filter(s=>s.winProb>=beThresh)
    : mode==="be5" ? data.filter(s=>s.winProb>=beThresh5)
    : mode==="posEV" ? data.filter(s=>s.ev>0)
    : data.slice(0, mode==="top10"?10:20);
  const posEV = data.filter(s=>s.ev>0);
  const avgEV = posEV.length>0 ? posEV.reduce((s,r)=>s+r.ev,0)/posEV.length : 0;

  return (
    <div>
      <div style={{display:"flex",gap:12,alignItems:"center",marginBottom:16,flexWrap:"wrap"}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:10,color:"#475569",textTransform:"uppercase",letterSpacing:0.5}}>Show</span>
          {[["posEV","+EV"],["be",`Win>${(beThresh*100).toFixed(0)}%`],["be5",`Win>${(beThresh5*100).toFixed(0)}%`],["top10","Top 10"],["top20","Top 20"]].map(([m,l])=>
            <Btn key={m} active={mode===m} onClick={()=>setMode(m)}>{l}</Btn>)}
        </div>
        <span style={{fontSize:11,color:"#334155"}}>
          {fmtHour(scanHour)} UTC — {posEV.length} positive-EV pairs
          {elapsed!=null&&` — ${elapsed}ms`}
          {modelWR10!=null&&` — val WR@10 ${(modelWR10*100).toFixed(0)}%`}
          {modelPnL10!=null&&` — val PnL@10 ${modelPnL10>0?"+":""}${modelPnL10}%`}
        </span>
      </div>

      {filtered.length===0 ? (
        <Box style={{padding:20,textAlign:"center",color:"#64748b",fontSize:12}}>
          No pairs meet the threshold at this scan hour.
        </Box>
      ) : (
        <Box style={{padding:12}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
            <span style={{fontSize:11,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase"}}>
              {filtered.length} pairs — TP +{activeTP}% / SL -{activeSL}% / {horizon}h horizon (break-even: {activeBE}% win rate)
            </span>
            {posEV.length>0&&<span style={{fontSize:11,color:"#22c55e"}}>Avg EV (positive): +{avgEV.toFixed(3)}%</span>}
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
              <thead><tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                {["#","Pair","Category","Price","Chg%","Win%","EV","Mom","RelVol","VWAP%","ATR","Vol","RSI","vsBTC","vsCat","Breadth","Gap"].map(h=>(
                  <th key={h} style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.5,textTransform:"uppercase",whiteSpace:"nowrap"}}>{h}</th>))}
              </tr></thead>
              <tbody>{filtered.map((s,i)=>{const chg=parseFloat(s.changeFromOpen);const evPos=s.ev>0;return(
                <tr key={s.ticker+s.rank} style={{borderBottom:"1px solid rgba(255,255,255,0.03)",
                  background:evPos?"rgba(34,197,94,0.03)":i%2?"rgba(255,255,255,0.015)":"transparent"}}
                  onMouseEnter={e=>e.currentTarget.style.background="rgba(255,255,255,0.04)"}
                  onMouseLeave={e=>e.currentTarget.style.background=evPos?"rgba(34,197,94,0.03)":i%2?"rgba(255,255,255,0.015)":"transparent"}>
                  <td style={{padding:"5px 6px",color:"#475569",fontWeight:600,fontSize:11}}>{s.rank}</td>
                  <td style={{padding:"5px 6px",fontWeight:700,color:"#e2e8f0",letterSpacing:0.3}}>{s.ticker}</td>
                  <td style={{padding:"5px 6px",color:"#64748b",fontSize:11}}>{s.sector}</td>
                  <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>${s.price}</td>
                  <td style={{padding:"5px 6px",color:chg>0?"#22c55e":chg<0?"#ef4444":"#94a3b8",fontVariantNumeric:"tabular-nums",fontWeight:500}}>{chg>0?"+":""}{chg}%</td>
                  <td style={{padding:"5px 6px"}} colSpan={2}><WinBar winProb={s.winProb} ev={s.ev} breakeven={beThresh}/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.momentum} label="momentum"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.relVolume} label="relVolume"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.vwapDist} label="vwapDist"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.atrReach} label="atrReach"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.realizedVol} label="realizedVol"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.rsi} label="rsi"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.retVsBtc||"—"} label="retVsBtc"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.retVsCat||"—"} label="retVsCat"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.catBreadth||"—"} label="catBreadth"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.gapPct||"—"} label="gapPct"/></td>
                </tr>);})}</tbody>
            </table>
          </div>
        </Box>
      )}
    </div>);
}

// ─── SWEEP (grid search over TP/SL) ──────────────────────────────
function SweepSection() {
  const [status,setStatus]=useState(null);
  const [results,setResults]=useState(null);

  const poll=useCallback(()=>{
    fetch('/api/sweep/status').then(r=>r.json()).then(setStatus).catch(()=>{});
    fetch('/api/sweep/results').then(r=>r.json()).then(setResults).catch(()=>{});
  },[]);
  useEffect(()=>{poll();const iv=setInterval(poll,3000);return()=>clearInterval(iv);},[poll]);

  const runSweep=async()=>{
    if(!confirm("Run grid search? 15 combinations, ~45-90 minutes. Scanner will be offline during this time. Runs in background — safe to close browser.")) return;
    await fetch('/api/sweep',{method:'POST'});
    poll();
  };
  const resetSweep=async()=>{
    if(!confirm("Discard all sweep results? The next sweep will start from scratch.")) return;
    await fetch('/api/sweep/reset',{method:'POST'});
    poll();
  };

  const ip = status?.inProgress;
  const grid = status?.grid;
  const gridResults = results?.grid || [];
  const resultMap = {};
  gridResults.forEach(r => { resultMap[`${r.tp_pct}_${r.sl_pct}`] = r; });
  const total = grid ? grid.tp.length * grid.sl.length : 0;
  const done = gridResults.length;

  const best = gridResults.filter(r=>r.avg_edge!=null).sort((a,b)=>b.avg_edge-a.avg_edge)[0];

  const edgeColor = (edge) => {
    if (edge == null) return "rgba(100,116,139,0.1)";
    if (edge >= 5) return "rgba(34,197,94,0.4)";
    if (edge >= 3) return "rgba(34,197,94,0.25)";
    if (edge >= 1) return "rgba(163,230,53,0.22)";
    if (edge >= 0) return "rgba(234,179,8,0.18)";
    if (edge >= -3) return "rgba(249,115,22,0.18)";
    return "rgba(239,68,68,0.22)";
  };

  return (
    <Box>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
        <Lbl>TP/SL Grid Search — 3 TP × 5 SL = 15 Combinations</Lbl>
        <div style={{display:"flex",gap:6}}>
          <Btn onClick={runSweep} disabled={ip} color="#f97316" style={{padding:"4px 10px",fontSize:11}}>
            {ip?`Running ${status?.current}/${status?.total}`:done>0&&done<total?`Resume (${done}/${total})`:"Run Sweep"}
          </Btn>
          {done>0&&!ip&&<Btn onClick={resetSweep} color="#ef4444" style={{padding:"4px 10px",fontSize:11}}>Reset</Btn>}
        </div>
      </div>

      {ip&&(
        <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(249,115,22,0.08)",border:"1px solid rgba(249,115,22,0.2)"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
            <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
              <div style={{width:`${(status.current/status.total)*100}%`,height:"100%",background:"#f97316",borderRadius:3,transition:"width 0.5s"}}/>
            </div>
            <span style={{fontSize:11,color:"#f97316",fontWeight:600}}>{status.current}/{status.total}</span>
          </div>
          <div style={{fontSize:11,color:"#94a3b8"}}>{status.message}</div>
        </div>
      )}

      {done===0&&!ip ? (
        <div style={{fontSize:12,color:"#64748b",padding:"20px 0",textAlign:"center"}}>
          Grid: TP {grid?.tp.join("%, ")||""}% × SL {grid?.sl.join("%, ")||""}%<br/>
          Each cell requires a full train cycle. First cell fetches bars (~20-30 min), subsequent use cache (~3-5 min).<br/>
          Safe to close browser — runs in background. Results save after each cell.
        </div>
      ) : (
        <>
          {best && (
            <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(34,197,94,0.06)",border:"1px solid rgba(34,197,94,0.2)"}}>
              <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:2}}>Best cell so far (by edge vs break-even)</div>
              <div style={{fontSize:13,color:"#e2e8f0"}}>
                <span style={{color:"#22c55e",fontWeight:700}}>TP {best.tp_pct}% / SL {best.sl_pct}%</span>
                <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                <span>Top-10 WR: <span style={{color:"#e2e8f0",fontWeight:600}}>{best.avg_top10_wr}%</span></span>
                <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                <span>Break-even: <span style={{color:"#eab308",fontWeight:600}}>{best.breakeven}%</span></span>
                <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                <span>Edge: <span style={{color:best.avg_edge>0?"#22c55e":"#ef4444",fontWeight:700}}>{best.avg_edge>0?"+":""}{best.avg_edge}%</span></span>
                <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                <span>Top-10 PnL: <span style={{color:best.avg_top10_pnl>0?"#22c55e":"#ef4444",fontWeight:600}}>{best.avg_top10_pnl>0?"+":""}{best.avg_top10_pnl}%</span></span>
              </div>
            </div>
          )}

          <div style={{marginBottom:8,fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase"}}>
            Edge = Top-10 Win Rate − Break-even (positive = tradable)
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{borderCollapse:"collapse",fontSize:11}}>
              <thead>
                <tr>
                  <th style={{padding:"6px 10px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>SL ↓ / TP →</th>
                  {grid?.tp.map(tp => (
                    <th key={tp} style={{padding:"6px 10px",textAlign:"center",color:"#94a3b8",fontSize:11,fontWeight:700}}>{tp}%</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {grid?.sl.map(sl => (
                  <tr key={sl}>
                    <td style={{padding:"6px 10px",textAlign:"right",color:"#94a3b8",fontWeight:700,fontSize:11}}>{sl}%</td>
                    {grid.tp.map(tp => {
                      const cell = resultMap[`${tp}_${sl}`];
                      const running = ip && status?.currentTP===tp && status?.currentSL===sl;
                      return (
                        <td key={tp} style={{
                          padding:"6px",border:"1px solid rgba(255,255,255,0.06)",
                          background:running?"rgba(249,115,22,0.2)":cell?edgeColor(cell.avg_edge):"rgba(100,116,139,0.05)",
                          minWidth:110,textAlign:"center"
                        }}>
                          {running ? (
                            <div style={{color:"#f97316",fontSize:10,fontWeight:600}}>Running...</div>
                          ) : cell ? (
                            <div>
                              <div style={{fontSize:14,fontWeight:700,color:cell.avg_edge>0?"#22c55e":cell.avg_edge<-3?"#ef4444":"#eab308",fontVariantNumeric:"tabular-nums"}}>
                                {cell.avg_edge>0?"+":""}{cell.avg_edge}%
                              </div>
                              <div style={{fontSize:9,color:"#94a3b8",marginTop:1}}>
                                WR {cell.avg_top10_wr}% / BE {cell.breakeven}%
                              </div>
                              <div style={{fontSize:9,color:cell.avg_top10_pnl>0?"#22c55e":"#ef4444",marginTop:1,fontVariantNumeric:"tabular-nums"}}>
                                PnL {cell.avg_top10_pnl>0?"+":""}{cell.avg_top10_pnl}%
                              </div>
                            </div>
                          ) : (
                            <div style={{color:"#334155",fontSize:10}}>—</div>
                          )}
                        </td>);
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{fontSize:10,color:"#475569",marginTop:8,lineHeight:1.5}}>
            Legend: green cells = model beats break-even. Warning: with 15 tests on the same validation slice, some positive results will be due to chance. Look for consistent patterns (e.g., a whole row or column trending positive) rather than single standout cells.
          </div>
        </>
      )}
    </Box>);
}

// ─── TRAINING ────────────────────────────────────────────────────
function TrainingTab() {
  const [d,setD]=useState(null);
  const [ld,setLd]=useState(true);
  const [sh,setSh]=useState(12);

  const poll=useCallback(()=>{fetch('/api/training/progress').then(r=>r.json()).then(d=>{setD(d);setLd(false);}).catch(()=>setLd(false));},[]);
  useEffect(()=>{poll();const iv=setInterval(poll,2000);return()=>clearInterval(iv);},[poll]);

  const [tp,setTp]=useState(2.0);
  const [sl,setSl]=useState(1.0);   // default matches server's SL_PCT; overridden by /api/health below
  // Initialize sliders from server's actual active TP/SL on mount. This
  // prevents the UI from silently overriding server defaults on first Train.
  useEffect(()=>{
    fetch('/api/health').then(r=>r.json()).then(h=>{
      if(typeof h?.tp_pct==='number') setTp(h.tp_pct);
      if(typeof h?.sl_pct==='number') setSl(h.sl_pct);
    }).catch(()=>{});
  },[]);
  const breakeven = (sl/(sl+tp)*100).toFixed(1);

  const trigTrain=async()=>{
    await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({tp_pct:tp,sl_pct:sl})});
    poll();
  };
  const clearCache=async()=>{
    if(!confirm("Clear cached bar data? Next training will refetch ~6 months from Coinbase (20-30 min).")) return;
    const r=await fetch('/api/cache/clear',{method:'POST'});
    const d=await r.json();
    alert(d.error?`Error: ${d.error}`:`Cleared: ${d.deleted.join(", ")||"nothing"}`);
  };

  if(ld) return <div style={{color:"#475569",padding:40,textAlign:"center"}}>Loading...</div>;
  const ip=d?.inProgress,pg=d||{},meta=d?.meta||{};
  const sm=meta[String(sh)];
  const activeTP = Object.values(meta)[0]?.tp_pct;
  const activeSL = Object.values(meta)[0]?.sl_pct;
  const activeBE = activeTP && activeSL ? (activeSL/(activeSL+activeTP)*100).toFixed(1) : null;

  const Slider = ({label,value,setValue,min,max,step,color})=>(
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
        <span style={{fontSize:11,color:"#94a3b8",fontWeight:500}}>{label}</span>
        <span style={{fontSize:13,color:color,fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{value.toFixed(2)}%</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e=>setValue(parseFloat(e.target.value))}
        disabled={ip}
        style={{width:"100%",accentColor:color,cursor:ip?"not-allowed":"pointer"}}/>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#475569",marginTop:2}}>
        <span>{min}%</span><span>{max}%</span>
      </div>
    </div>);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <Lbl>Model Training — First-Passage</Lbl>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
          <div>
            <div style={{fontSize:11,color:"#64748b",marginBottom:10,textTransform:"uppercase",letterSpacing:0.5}}>
              Strategy Parameters
            </div>
            <Slider label="Take Profit" value={tp} setValue={setTp} min={0.50} max={5.00} step={0.10} color="#22c55e"/>
            <Slider label="Stop Loss" value={sl} setValue={setSl} min={0.50} max={6.00} step={0.10} color="#ef4444"/>
            <div style={{marginTop:12,padding:"8px 10px",borderRadius:4,
              background:"rgba(234,179,8,0.06)",border:"1px solid rgba(234,179,8,0.15)"}}>
              <div style={{fontSize:11,color:"#94a3b8",marginBottom:2}}>Break-even win rate (this setting)</div>
              <div style={{fontSize:18,color:"#eab308",fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{breakeven}%</div>
              <div style={{fontSize:10,color:"#64748b",marginTop:2}}>
                Model must exceed this to be profitable. Reward:Risk = {(tp/sl).toFixed(2)}:1
              </div>
            </div>
          </div>
          <div>
            <div style={{fontSize:11,color:"#64748b",marginBottom:10,textTransform:"uppercase",letterSpacing:0.5}}>
              Current Status
            </div>
            <div style={{fontSize:12,lineHeight:2,color:"#94a3b8",marginBottom:14}}>
              {[
                {l:"Models",v:Object.keys(meta).length>0?`${Object.keys(meta).length} hours trained`:"None",ok:Object.keys(meta).length>0},
                {l:"Active TP/SL",v:activeTP?`+${activeTP}% / -${activeSL}%`:"—",ok:!!activeTP},
                {l:"Active break-even",v:activeBE?`${activeBE}%`:"—",ok:!!activeBE},
                {l:"Trained",v:Object.values(meta)[0]?.trained_at?new Date(Object.values(meta)[0].trained_at).toLocaleString():"Never",ok:Object.keys(meta).length>0},
              ].map((c,i)=>(
                <div key={i} style={{display:"flex",alignItems:"center",gap:8}}>
                  <span style={{width:12,height:12,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
                    background:c.ok?"rgba(34,197,94,0.15)":"rgba(100,116,139,0.15)",color:c.ok?"#22c55e":"#64748b",fontSize:9,fontWeight:900}}>{c.ok?"✓":"·"}</span>
                  <span style={{minWidth:130,fontSize:11}}>{c.l}</span>
                  <span style={{color:c.ok?"#e2e8f0":"#64748b",fontWeight:500}}>{c.v}</span>
                </div>))}
            </div>
            <div style={{display:"flex",gap:8}}>
              <Btn onClick={trigTrain} disabled={ip} color="#8b5cf6" style={{padding:"8px 16px",fontSize:12}}>
                {ip?"Training...":`Train (TP ${tp}% / SL ${sl}%)`}
              </Btn>
              <Btn onClick={clearCache} disabled={ip} color="#ef4444" style={{padding:"8px 12px",fontSize:11}}>
                Clear cache
              </Btn>
            </div>
            {ip&&(
              <div style={{marginTop:10}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                  <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                    <div style={{width:`${pg.pct||0}%`,height:"100%",background:"#8b5cf6",borderRadius:3,transition:"width 0.5s"}}/>
                  </div>
                  <span style={{fontSize:11,color:"#8b5cf6",fontWeight:600}}>{pg.pct||0}%</span>
                </div>
                <div style={{fontSize:11,color:"#64748b"}}>{pg.message}</div>
              </div>)}
            <div style={{marginTop:10,fontSize:10,color:"#475569",lineHeight:1.5}}>
              First training fetches ~6 months of 15m bars for ~140 pairs (~20-30 min). Re-training with different TP/SL uses cached bars (~3-5 min).
            </div>
          </div>
        </div>
      </Box>

      {Object.keys(meta).length>0&&(
        <Box>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
            <Lbl>Validation Results</Lbl>
            <div style={{display:"flex",gap:4,marginBottom:10}}>
              {SCAN_HOURS.filter(h=>meta[String(h)]).map(h=><Btn key={h} active={h===sh} onClick={()=>setSh(h)}>{fmtHour(h)}</Btn>)}
            </div>
          </div>
          {sm?(
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              <div>
                <div style={{fontSize:11,color:"#64748b",marginBottom:8,textTransform:"uppercase",letterSpacing:0.5}}>Key Metrics</div>
                <div style={{fontSize:12,lineHeight:2.2,color:"#94a3b8"}}>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>AUC:</span>
                    <span style={{color:sm.auc>0.6?"#22c55e":"#eab308",fontWeight:700,fontSize:14}}>{sm.auc}</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Top-10 Win Rate:</span>
                    <span style={{color:sm.avg_win_rate_top10>sm.val_win_rate?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{(sm.avg_win_rate_top10*100).toFixed(1)}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>vs base {(sm.val_win_rate*100).toFixed(1)}%</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Top-10 Avg P&L:</span>
                    <span style={{color:sm.avg_pnl_top10>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.avg_pnl_top10>0?"+":""}{sm.avg_pnl_top10}%</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Win&gt;BE pairs):</span>
                    <span style={{color:sm.ev_above_breakeven>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_breakeven>0?"+":""}{sm.ev_above_breakeven}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_breakeven} pairs)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Win&gt;BE+5% pairs):</span>
                    <span style={{color:sm.ev_above_breakeven_plus5>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_breakeven_plus5>0?"+":""}{sm.ev_above_breakeven_plus5}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_breakeven_plus5} pairs)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Top-10 default):</span>
                    <span style={{color:sm.ev_above_50pct>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_50pct>0?"+":""}{sm.ev_above_50pct}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_50pct} samples @ &gt;50%)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Exit reasons (val):</span>
                    <span style={{fontSize:11}}>{sm.val_exit_reasons?Object.entries(sm.val_exit_reasons).map(([r,n])=>`${r}: ${n}`).join(", "):""}</span></div>
                </div>
              </div>
              <div>
                <div style={{fontSize:11,color:"#64748b",marginBottom:8,textTransform:"uppercase",letterSpacing:0.5}}>Feature Importance</div>
                {sm.importance&&Object.entries(sm.importance).sort(([,a],[,b])=>b-a).slice(0,12).map(([name,val])=>{
                  const max=Math.max(...Object.values(sm.importance));
                  return (
                    <div key={name} style={{display:"flex",alignItems:"center",gap:8,marginBottom:3}}>
                      <span style={{width:100,fontSize:10,color:"#94a3b8",textAlign:"right",flexShrink:0}}>{name}</span>
                      <div style={{flex:1,height:12,background:"rgba(255,255,255,0.04)",borderRadius:2,overflow:"hidden"}}>
                        <div style={{width:`${(val/max)*100}%`,height:"100%",borderRadius:2,background:"#8b5cf6"}}/>
                      </div>
                      <span style={{fontSize:10,color:"#64748b",minWidth:32,fontVariantNumeric:"tabular-nums"}}>{(val*100).toFixed(1)}%</span>
                    </div>);})}
              </div>
            </div>
          ):<div style={{color:"#475569",fontSize:12}}>Select scan hour</div>}
        </Box>)}
      <SweepSection/>
    </div>);
}

// ─── OUTCOMES ────────────────────────────────────────────────────
function OutcomesTab() {
  const [d,setD]=useState(null);
  useEffect(()=>{fetch('/api/outcomes/summary').then(r=>r.json()).then(setD).catch(()=>{});},[]);
  if(!d) return <div style={{color:"#475569",padding:40,textAlign:"center"}}>Loading...</div>;
  if(d.totalDays===0) return <Box><div style={{color:"#475569",fontSize:12,padding:20,textAlign:"center"}}>No outcomes yet. Recorded daily at 00:30 UTC.</div></Box>;
  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <Lbl>Top-10 Win Rate & P&L — {d.totalDays} days</Lbl>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
            <thead><tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
              <th style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10}}>DATE</th>
              {SCAN_HOURS.map(h=><th key={h} style={{padding:"6px",textAlign:"center",color:"#64748b",fontSize:10}}>{fmtHour(h)}</th>)}
            </tr></thead>
            <tbody>{d.recent.map((day,i)=>(
              <tr key={i} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{day.date}</td>
                {SCAN_HOURS.map(h=>{
                  const s=day.hours[String(h)];
                  if(!s) return <td key={h} style={{padding:"5px 6px",textAlign:"center",color:"#334155"}}>—</td>;
                  const wr=s.top10wins*10;
                  const pnl=s.top10pnl||0;
                  const bwr=s.baseWR||0;
                  const c=wr>bwr+10?"#22c55e":wr>bwr?"#eab308":"#ef4444";
                  const pc=pnl>0?"#22c55e":"#ef4444";
                  return <td key={h} style={{padding:"5px 6px",textAlign:"center",fontVariantNumeric:"tabular-nums"}}>
                    <span style={{color:c,fontWeight:600}}>{wr}%</span>
                    <span style={{fontSize:10,color:"#475569",marginLeft:3}}>({s.top10wins}/10)</span>
                    <div style={{fontSize:10,color:pc,fontWeight:500}}>{pnl>0?"+":""}{pnl}%</div>
                    <div style={{fontSize:9,color:"#334155"}}>base {bwr}%</div>
                  </td>;
                })}
              </tr>))}</tbody>
          </table>
        </div>
      </Box>
    </div>);
}

// ─── STATUS ──────────────────────────────────────────────────────
function StatusTab({health}) {
  return <Box><Lbl>Server</Lbl>
    {health?<div style={{fontSize:12,lineHeight:2,color:"#94a3b8"}}>
      {[
        {l:"Server",v:"Online",ok:true},
        {l:"Coinbase API",v:"Public (no auth)",ok:true},
        {l:"Market",v:"24/7 — always open",ok:true},
        {l:"Universe",v:health.universeSize?`${health.universeSize} pairs`:"—",ok:(health.universeSize||0)>0},
        {l:"Benchmark",v:health.benchmark||"BTC-USD",ok:true},
        {l:"Models",v:health.modelsLoaded?.length>0?health.modelsLoaded.join(", "):"None",ok:health.modelsLoaded?.length>0},
        {l:"Horizon",v:health.horizonHours?`${health.horizonHours}h per trade`:"—",ok:!!health.horizonHours},
        {l:"Strategy",v:`TP +${health.tp_pct}% / SL -${health.sl_pct}%`,ok:true},
        {l:"Outcome days",v:String(health.outcomeDays||0),ok:(health.outcomeDays||0)>0},
        {l:"Cached scans",v:health.lastScanHours?.length>0?health.lastScanHours.join(", "):"None",ok:health.hasLastScan},
      ].map((c,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{width:14,height:14,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
            background:c.ok?"rgba(34,197,94,0.15)":"rgba(239,68,68,0.15)",color:c.ok?"#22c55e":"#ef4444",fontSize:10,fontWeight:900}}>{c.ok?"✓":"✗"}</span>
          <span style={{minWidth:140}}>{c.l}</span>
          <span style={{color:c.ok?"#e2e8f0":"#ef4444",fontWeight:500}}>{c.v}</span>
        </div>))}
    </div>:<div style={{color:"#ef4444"}}>Cannot reach server</div>}
  </Box>;
}

// ─── V2 ──────────────────────────────────────────────────────────
// Stage 1 UI: trigger training, watch progress, list trained cells, download diagnostic.
// Preset cells let you click to train without typing anything.
const V2_PRESETS = [
  { k_atr: 0.5, horizon_hours: 2,  label: "k=0.5 / 2h  (low bar, fast)"    },
  { k_atr: 1.0, horizon_hours: 4,  label: "k=1.0 / 4h  (baseline)"         },
  { k_atr: 1.5, horizon_hours: 8,  label: "k=1.5 / 8h  (moderate)"         },
  { k_atr: 2.0, horizon_hours: 12, label: "k=2.0 / 12h (high bar, slow)"   },
  { k_atr: 2.5, horizon_hours: 24, label: "k=2.5 / 24h (high bar, 1-day)"  },
];

function V2Tab() {
  const [progress, setProgress] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedCell, setSelectedCell] = useState(null);
  const [cellDetail, setCellDetail] = useState(null);
  const [customK, setCustomK] = useState(1.0);
  const [customH, setCustomH] = useState(4);
  const [error, setError] = useState(null);

  // Poll progress + models every 3s
  useEffect(() => {
    const fetchAll = () => {
      fetch('/api/v2/training/progress').then(r => r.json()).then(setProgress).catch(() => {});
      fetch('/api/v2/models').then(r => r.json()).then(d => setModels(d.models || [])).catch(() => {});
    };
    fetchAll();
    const iv = setInterval(fetchAll, 3000);
    return () => clearInterval(iv);
  }, []);

  const trainCell = useCallback(async (k_atr, horizon_hours) => {
    setError(null);
    try {
      const r = await fetch('/api/v2/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({k_atr, horizon_hours}),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
      if (d.status === "already_running") {
        setError(`Training already in progress — wait for current cell to finish`);
        return;
      }
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const fetchCellDetail = useCallback(async (k_atr, horizon_hours) => {
    setSelectedCell({k_atr, horizon_hours});
    setCellDetail(null);
    try {
      const r = await fetch(`/api/v2/model/${k_atr}/${horizon_hours}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setCellDetail(await r.json());
    } catch (e) {
      setCellDetail({error: e.message});
    }
  }, []);

  const downloadDiag = useCallback(async () => {
    try {
      const r = await fetch('/api/v2/diagnostic');
      const b = await r.blob();
      const fn = r.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1] || 'v2_diag.json';
      const u = URL.createObjectURL(b);
      const a = document.createElement('a');
      a.href = u; a.download = fn;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(u);
    } catch (e) { alert(e.message); }
  }, []);

  const ip = progress?.inProgress;
  const phase = progress?.phase;

  // Color-code precision: green ≥0.70, yellow 0.55-0.70, red <0.55
  const precColor = (p) => {
    if (p == null) return "#64748b";
    if (p >= 0.70) return "#22c55e";
    if (p >= 0.55) return "#eab308";
    return "#ef4444";
  };

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>

      {/* ── BANNER ── */}
      <Box>
        <div style={{fontSize:11,color:"#8b5cf6",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6,fontWeight:700}}>
          v2 Stage 1 — Vol-Normalized Threshold Classifier
        </div>
        <div style={{fontSize:12,color:"#94a3b8",lineHeight:1.6}}>
          Asks: "will price touch <span style={{color:"#e2e8f0"}}>+k × coin's 7-day ATR</span> at any point within <span style={{color:"#e2e8f0"}}>H hours</span>?"
          Each (k, H) combination is a separate model — train several and compare precision at the 0.75 threshold.
          A cell is <span style={{color:"#22c55e"}}>productive</span> if precision @ 0.75 ≥ 70% with ≥1 prediction/day.
        </div>
      </Box>

      {/* ── PROGRESS ── */}
      <Box>
        <Lbl>Training Progress</Lbl>
        {!progress ? (
          <div style={{fontSize:12,color:"#475569"}}>Loading...</div>
        ) : ip ? (
          <div>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${progress.pct||0}%`,height:"100%",background:"#8b5cf6",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#8b5cf6",fontWeight:600,minWidth:36,textAlign:"right"}}>{progress.pct||0}%</span>
            </div>
            <div style={{fontSize:12,color:"#94a3b8"}}>{progress.message}</div>
            {progress.cell && (
              <div style={{fontSize:11,color:"#64748b",marginTop:4}}>
                Training cell: k={progress.cell.k_atr}, H={progress.cell.horizon_hours}h
              </div>
            )}
          </div>
        ) : phase === "done" ? (
          <div style={{fontSize:12,color:"#22c55e"}}>✓ {progress.message}</div>
        ) : phase === "error" ? (
          <div style={{fontSize:12,color:"#ef4444"}}>✗ {progress.message}</div>
        ) : (
          <div style={{fontSize:12,color:"#475569"}}>Idle — click a preset below to start training.</div>
        )}
      </Box>

      {error && (
        <div style={{padding:"8px 12px",borderRadius:6,background:"rgba(239,68,68,0.1)",border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:12}}>
          {error}
        </div>
      )}

      {/* ── TRAIN BUTTONS ── */}
      <Box>
        <Lbl>Train a Cell</Lbl>
        <div style={{fontSize:11,color:"#64748b",marginBottom:10}}>
          Each cell is an independent experiment. Click a preset or enter custom values.
          Training reuses cached bars from v1 (if fresh) → ~3-5 min; otherwise refetches → ~20-30 min.
        </div>

        <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Presets</div>
        <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:16}}>
          {V2_PRESETS.map(p => (
            <Btn
              key={`${p.k_atr}_${p.horizon_hours}`}
              onClick={() => trainCell(p.k_atr, p.horizon_hours)}
              disabled={ip}
              color="#8b5cf6"
              style={{padding:"6px 12px",fontSize:11}}
            >
              {p.label}
            </Btn>
          ))}
        </div>

        <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Custom</div>
        <div style={{display:"flex",gap:12,alignItems:"center",flexWrap:"wrap"}}>
          <label style={{fontSize:11,color:"#94a3b8",display:"flex",flexDirection:"column",gap:4}}>
            <span>k_atr (0.1–10)</span>
            <input type="number" step="0.1" min="0.1" max="10" value={customK}
              onChange={e=>setCustomK(parseFloat(e.target.value)||1.0)} disabled={ip}
              style={{width:70,padding:"4px 6px",background:"rgba(255,255,255,0.04)",border:"1px solid rgba(255,255,255,0.08)",borderRadius:4,color:"#e2e8f0",fontFamily:F,fontSize:12}}/>
          </label>
          <label style={{fontSize:11,color:"#94a3b8",display:"flex",flexDirection:"column",gap:4}}>
            <span>horizon_hours (1–72)</span>
            <input type="number" step="1" min="1" max="72" value={customH}
              onChange={e=>setCustomH(parseInt(e.target.value)||4)} disabled={ip}
              style={{width:70,padding:"4px 6px",background:"rgba(255,255,255,0.04)",border:"1px solid rgba(255,255,255,0.08)",borderRadius:4,color:"#e2e8f0",fontFamily:F,fontSize:12}}/>
          </label>
          <Btn
            onClick={() => trainCell(customK, customH)}
            disabled={ip}
            color="#8b5cf6"
            style={{padding:"6px 14px",fontSize:11,marginTop:18}}
          >
            Train k={customK} / {customH}h
          </Btn>
        </div>
      </Box>

      {/* ── TRAINED CELLS ── */}
      <Box>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
          <Lbl>Trained Cells ({models.length})</Lbl>
          {models.length > 0 && (
            <Btn onClick={downloadDiag} color="#f97316" style={{padding:"4px 10px",fontSize:11}}>
              ⬇ Download all (v2 diagnostic)
            </Btn>
          )}
        </div>
        {models.length === 0 ? (
          <div style={{fontSize:12,color:"#475569",padding:"20px 0",textAlign:"center"}}>
            No cells trained yet. Click a preset above to start.
          </div>
        ) : (
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
              <thead>
                <tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                  {["Cell","AUC (test)","Base rate","Prec @ 0.75","N @ 0.75","Per day","Iter","Trained"].map(h => (
                    <th key={h} style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.5,textTransform:"uppercase"}}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models
                  .slice()
                  .sort((a,b) => (b.precision_at_0_75 ?? 0) - (a.precision_at_0_75 ?? 0))
                  .map(m => {
                  const sel = selectedCell && selectedCell.k_atr === m.k_atr && selectedCell.horizon_hours === m.horizon_hours;
                  const aucC = m.auc_test >= 0.65 ? "#22c55e" : m.auc_test >= 0.55 ? "#eab308" : "#ef4444";
                  return (
                    <tr key={`${m.k_atr}_${m.horizon_hours}`}
                      onClick={() => fetchCellDetail(m.k_atr, m.horizon_hours)}
                      style={{borderBottom:"1px solid rgba(255,255,255,0.03)",cursor:"pointer",
                        background:sel?"rgba(139,92,246,0.08)":"transparent"}}>
                      <td style={{padding:"6px 8px",color:"#e2e8f0",fontWeight:700,fontVariantNumeric:"tabular-nums"}}>
                        k={m.k_atr} / {m.horizon_hours}h
                      </td>
                      <td style={{padding:"6px 8px",fontVariantNumeric:"tabular-nums",color:aucC,fontWeight:600}}>
                        {m.auc_test?.toFixed(3) ?? "—"}
                      </td>
                      <td style={{padding:"6px 8px",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                        {m.base_rate_test != null ? `${(m.base_rate_test*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"6px 8px",fontVariantNumeric:"tabular-nums",color:precColor(m.precision_at_0_75),fontWeight:700}}>
                        {m.precision_at_0_75 != null ? `${(m.precision_at_0_75*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"6px 8px",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                        {m.n_at_0_75 ?? 0}
                      </td>
                      <td style={{padding:"6px 8px",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                        {m.per_day_at_0_75?.toFixed(1) ?? "0"}
                      </td>
                      <td style={{padding:"6px 8px",fontVariantNumeric:"tabular-nums",color:"#64748b"}}>
                        {m.best_iteration ?? "—"}
                      </td>
                      <td style={{padding:"6px 8px",fontSize:10,color:"#475569"}}>
                        {m.trained_at ? new Date(m.trained_at).toLocaleString() : "—"}
                      </td>
                    </tr>);
                })}
              </tbody>
            </table>
          </div>
        )}
      </Box>

      {/* ── CELL DETAIL ── */}
      {selectedCell && (
        <Box>
          <Lbl>Cell Detail — k={selectedCell.k_atr} / {selectedCell.horizon_hours}h</Lbl>
          {!cellDetail ? (
            <div style={{color:"#475569",fontSize:12}}>Loading...</div>
          ) : cellDetail.error ? (
            <div style={{color:"#ef4444",fontSize:12}}>{cellDetail.error}</div>
          ) : (
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              <div>
                <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:8}}>Precision by confidence threshold</div>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                  <thead>
                    <tr style={{borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
                      <th style={{padding:"4px 6px",textAlign:"left",color:"#64748b",fontSize:10}}>Threshold</th>
                      <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Precision</th>
                      <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>N</th>
                      <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Per day</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cellDetail.precision_at_threshold && Object.entries(cellDetail.precision_at_threshold).map(([t, d]) => (
                      <tr key={t} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                        <td style={{padding:"4px 6px",fontVariantNumeric:"tabular-nums"}}>{t}</td>
                        <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(d.precision),fontWeight:600}}>
                          {d.n_predictions > 0 ? `${(d.precision*100).toFixed(1)}%` : "—"}
                        </td>
                        <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                          {d.n_predictions}
                        </td>
                        <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                          {d.avg_per_day?.toFixed(1) ?? "0"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{marginTop:16}}>
                  <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6}}>Top-K precision</div>
                  <div style={{fontSize:11,color:"#94a3b8",lineHeight:1.8}}>
                    {cellDetail.top_k_precision && Object.entries(cellDetail.top_k_precision).map(([k, d]) => (
                      <div key={k}><span style={{color:"#64748b",display:"inline-block",minWidth:110}}>{k.replace(/_/g," ")}</span>
                        <span style={{color:precColor(d.precision),fontWeight:600,marginLeft:8}}>
                          {(d.precision*100).toFixed(1)}%
                        </span>
                        <span style={{color:"#475569",marginLeft:6,fontSize:10}}>(n={d.n})</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div style={{marginTop:16,fontSize:11,color:"#94a3b8",lineHeight:1.8}}>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:110}}>Train/Val/Test rows:</span>
                    <span style={{fontVariantNumeric:"tabular-nums"}}>{cellDetail.train_rows} / {cellDetail.val_rows} / {cellDetail.test_rows}</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:110}}>Embargo:</span>
                    <span style={{fontVariantNumeric:"tabular-nums"}}>{cellDetail.embargo_days} days</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:110}}>Best iter:</span>
                    <span style={{fontVariantNumeric:"tabular-nums"}}>{cellDetail.best_iteration}</span></div>
                </div>
              </div>
              <div>
                <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:8}}>Feature importance (top 15)</div>
                {cellDetail.feature_importance && Object.entries(cellDetail.feature_importance)
                  .sort(([,a],[,b]) => b-a).slice(0,15).map(([name, val]) => {
                    const allVals = Object.values(cellDetail.feature_importance);
                    const max = Math.max(...allVals);
                    return (
                      <div key={name} style={{display:"flex",alignItems:"center",gap:8,marginBottom:3}}>
                        <span style={{width:150,fontSize:10,color:"#94a3b8",textAlign:"right",flexShrink:0,fontFamily:F}}>{name}</span>
                        <div style={{flex:1,height:12,background:"rgba(255,255,255,0.04)",borderRadius:2,overflow:"hidden"}}>
                          <div style={{width:`${(val/max)*100}%`,height:"100%",borderRadius:2,background:"#8b5cf6"}}/>
                        </div>
                        <span style={{fontSize:10,color:"#64748b",minWidth:36,textAlign:"right",fontVariantNumeric:"tabular-nums"}}>
                          {(val*100).toFixed(1)}%
                        </span>
                      </div>);
                  })}
              </div>
            </div>
          )}
        </Box>
      )}
    </div>
  );
}

// ─── RULES (v2 Stage 2) ──────────────────────────────────────────
// Rule mining UI. Three parts:
//   1. Mine panel — configure & trigger a rule mining run
//   2. Catalogs list — one per (threshold, horizon); click to load its rules
//   3. Rules table + detail — sortable/paginated; click a row to see full detail
//
// Rule mining is always scoped to ONE threshold at a time. Within a run, each
// horizon (4h/6h/8h by default) gets an independent catalog. Cross-horizon
// retest is annotated inside each rule so the user can see if a rule found
// on h=4 also holds at h=6 or h=8.

function RulesTab() {
  // Mine panel state
  const [thresholdPct, setThresholdPct] = useState(2.0);   // percent (not fraction)
  const [horizons, setHorizons] = useState({4:true, 6:true, 8:true});
  const [methods, setMethods] = useState({univariate:true, tree:true, apriori:true});
  const [minPrec, setMinPrec] = useState(0.65);
  const [minSup, setMinSup] = useState(30);
  const [minLift, setMinLift] = useState(0.05);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);

  // Catalogs list
  const [catalogs, setCatalogs] = useState([]);
  const [selectedCatalog, setSelectedCatalog] = useState(null);  // {threshold_bps, horizon_hours}
  const [catalogDetail, setCatalogDetail] = useState(null);

  // Rules table
  const [sortKey, setSortKey] = useState("test_precision");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);
  const [pageSize] = useState(25);
  const [selectedRuleId, setSelectedRuleId] = useState(null);
  const [ruleDetail, setRuleDetail] = useState(null);

  // Poll progress + catalogs
  useEffect(() => {
    const fetchState = () => {
      fetch('/api/v2/mine_rules/progress').then(r => r.json()).then(setProgress).catch(() => {});
      fetch('/api/v2/rules/catalogs').then(r => r.json())
        .then(d => setCatalogs(d.catalogs || [])).catch(() => {});
    };
    fetchState();
    const iv = setInterval(fetchState, 3000);
    return () => clearInterval(iv);
  }, []);

  // Load catalog detail when selected
  useEffect(() => {
    if (!selectedCatalog) return setCatalogDetail(null);
    const {threshold_bps, horizon_hours} = selectedCatalog;
    setCatalogDetail(null);
    setPage(0);
    setSelectedRuleId(null);
    setRuleDetail(null);
    fetch(`/api/v2/rules/catalog/${threshold_bps}/${horizon_hours}`)
      .then(r => r.json()).then(setCatalogDetail).catch(e => setCatalogDetail({error: e.message}));
  }, [selectedCatalog]);

  // Load rule detail when selected
  useEffect(() => {
    if (!selectedRuleId || !selectedCatalog) return setRuleDetail(null);
    const {threshold_bps, horizon_hours} = selectedCatalog;
    setRuleDetail(null);
    fetch(`/api/v2/rules/rule/${threshold_bps}/${horizon_hours}/${selectedRuleId}`)
      .then(r => r.json()).then(setRuleDetail).catch(e => setRuleDetail({error: e.message}));
  }, [selectedRuleId, selectedCatalog]);

  const startMining = useCallback(async () => {
    setError(null);
    const horizonList = Object.keys(horizons).filter(h => horizons[h]).map(h => parseInt(h));
    const methodList = Object.keys(methods).filter(m => methods[m]);
    if (horizonList.length === 0) { setError("Pick at least one horizon"); return; }
    if (methodList.length === 0) { setError("Pick at least one method"); return; }
    try {
      const r = await fetch('/api/v2/mine_rules', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          threshold_pct: thresholdPct / 100,
          horizon_hours: horizonList,
          methods: methodList,
          min_precision: minPrec,
          min_support: minSup,
          min_lift: minLift,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
      if (d.status === "already_running") setError("Mining already in progress");
    } catch (e) { setError(e.message); }
  }, [thresholdPct, horizons, methods, minPrec, minSup, minLift]);

  const deleteCatalog = useCallback(async (threshold_bps, horizon_hours) => {
    if (!confirm(`Delete catalog for threshold_bps=${threshold_bps}, horizon=${horizon_hours}h?`)) return;
    try {
      const r = await fetch(`/api/v2/rules/catalog/${threshold_bps}/${horizon_hours}`, {method: 'DELETE'});
      if (!r.ok) { const d = await r.json(); throw new Error(d.error); }
      // Refresh catalogs list
      const rr = await fetch('/api/v2/rules/catalogs'); const dd = await rr.json();
      setCatalogs(dd.catalogs || []);
      if (selectedCatalog && selectedCatalog.threshold_bps === threshold_bps &&
          selectedCatalog.horizon_hours === horizon_hours) {
        setSelectedCatalog(null);
      }
    } catch (e) { alert(e.message); }
  }, [selectedCatalog]);

  const downloadDiagnostic = useCallback(async () => {
    try {
      const r = await fetch('/api/v2/rules/diagnostic');
      const b = await r.blob();
      const fn = r.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1] || 'rules_diag.json';
      const u = URL.createObjectURL(b);
      const a = document.createElement('a');
      a.href = u; a.download = fn;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(u);
    } catch (e) { alert(e.message); }
  }, []);

  const ip = progress?.inProgress;
  const phase = progress?.phase;

  // Color code precision: green >= 0.70, yellow 0.55-0.70, red < 0.55
  const precColor = (p) => {
    if (p == null) return "#64748b";
    if (p >= 0.70) return "#22c55e";
    if (p >= 0.55) return "#eab308";
    return "#ef4444";
  };

  // Sort/paginate rules
  const rules = catalogDetail?.rules || [];
  const sortedRules = [...rules].sort((a, b) => {
    let av, bv;
    if (sortKey === "test_precision") { av = a.test?.precision ?? 0; bv = b.test?.precision ?? 0; }
    else if (sortKey === "train_precision") { av = a.train?.precision ?? 0; bv = b.train?.precision ?? 0; }
    else if (sortKey === "test_support") { av = a.test?.support ?? 0; bv = b.test?.support ?? 0; }
    else if (sortKey === "lift") { av = a.test?.lift_vs_base ?? 0; bv = b.test?.lift_vs_base ?? 0; }
    else if (sortKey === "gap") { av = a.train_test_gap ?? 0; bv = b.train_test_gap ?? 0; }
    else if (sortKey === "n_conds") { av = a.conditions?.length ?? 0; bv = b.conditions?.length ?? 0; }
    else { av = 0; bv = 0; }
    return sortDir === "desc" ? (bv - av) : (av - bv);
  });
  const pageStart = page * pageSize;
  const pageEnd = Math.min(pageStart + pageSize, sortedRules.length);
  const pageRules = sortedRules.slice(pageStart, pageEnd);
  const totalPages = Math.ceil(sortedRules.length / pageSize);

  const clickHeader = (k) => {
    if (sortKey === k) setSortDir(sortDir === "desc" ? "asc" : "desc");
    else { setSortKey(k); setSortDir("desc"); }
    setPage(0);
  };

  // Compact rule-condition display for table row
  const condSummary = (r) => {
    const n = r.conditions?.length ?? 0;
    const feats = (r.conditions || []).map(c => c.feature).slice(0, 3);
    return `${n} cond: ${feats.join(", ")}${n > 3 ? "..." : ""}`;
  };

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>

      {/* ── BANNER ── */}
      <Box>
        <div style={{fontSize:11,color:"#8b5cf6",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6,fontWeight:700}}>
          v2 Stage 2 — Rule Mining
        </div>
        <div style={{fontSize:12,color:"#94a3b8",lineHeight:1.6}}>
          Mines interpretable setups from historical data. Three methods run in parallel per horizon:{" "}
          <span style={{color:"#e2e8f0"}}>univariate</span> (single-feature thresholds),{" "}
          <span style={{color:"#e2e8f0"}}>decision trees</span> (multi-feature, precision-optimized leaves),{" "}
          <span style={{color:"#e2e8f0"}}>apriori</span> (frequent feature-bin combinations).
          Each horizon gets its own catalog. Every rule is cross-horizon retested to see if it generalizes.
          Absolute +X% threshold (not vol-normalized).
        </div>
      </Box>

      {/* ── MINE PANEL ── */}
      <Box>
        <Lbl>Mine Rules</Lbl>

        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit, minmax(240px, 1fr))",gap:16,marginBottom:12}}>
          {/* Threshold */}
          <div>
            <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6}}>Threshold (+%)</div>
            <div style={{display:"flex",gap:8,alignItems:"center"}}>
              <input type="number" step="0.1" min="0.5" max="20" value={thresholdPct}
                onChange={e => setThresholdPct(parseFloat(e.target.value) || 2.0)}
                disabled={ip}
                style={{width:70,padding:"4px 6px",background:"rgba(255,255,255,0.04)",border:"1px solid rgba(255,255,255,0.08)",borderRadius:4,color:"#e2e8f0",fontFamily:F,fontSize:12}}/>
              <span style={{fontSize:11,color:"#475569"}}>% (e.g., 2.0 = +2%)</span>
            </div>
          </div>

          {/* Horizons */}
          <div>
            <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6}}>Horizons (hours)</div>
            <div style={{display:"flex",gap:10}}>
              {[2,4,6,8,12,24].map(h => (
                <label key={h} style={{display:"flex",alignItems:"center",gap:4,fontSize:12,cursor:ip?"not-allowed":"pointer",opacity:ip?0.5:1}}>
                  <input type="checkbox" checked={!!horizons[h]}
                    onChange={e => setHorizons(s => ({...s, [h]: e.target.checked}))}
                    disabled={ip}/>
                  <span style={{color:horizons[h]?"#e2e8f0":"#64748b"}}>{h}h</span>
                </label>
              ))}
            </div>
          </div>

          {/* Methods */}
          <div>
            <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6}}>Methods</div>
            <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
              {["univariate","tree","apriori"].map(m => (
                <label key={m} style={{display:"flex",alignItems:"center",gap:4,fontSize:12,cursor:ip?"not-allowed":"pointer",opacity:ip?0.5:1}}>
                  <input type="checkbox" checked={!!methods[m]}
                    onChange={e => setMethods(s => ({...s, [m]: e.target.checked}))}
                    disabled={ip}/>
                  <span style={{color:methods[m]?"#e2e8f0":"#64748b"}}>{m}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit, minmax(220px, 1fr))",gap:16,marginBottom:12}}>
          {/* min precision */}
          <div>
            <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6}}>
              Min precision: <span style={{color:"#e2e8f0"}}>{(minPrec*100).toFixed(0)}%</span>
            </div>
            <input type="range" min="0.3" max="0.95" step="0.01" value={minPrec}
              onChange={e => setMinPrec(parseFloat(e.target.value))} disabled={ip}
              style={{width:"100%"}}/>
            <div style={{fontSize:10,color:"#475569",marginTop:2}}>Rules below this are discarded</div>
          </div>

          {/* min support */}
          <div>
            <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6}}>
              Min support: <span style={{color:"#e2e8f0"}}>{minSup}</span>
            </div>
            <input type="number" step="1" min="5" max="5000" value={minSup}
              onChange={e => setMinSup(parseInt(e.target.value) || 30)} disabled={ip}
              style={{width:"100%",padding:"4px 6px",background:"rgba(255,255,255,0.04)",border:"1px solid rgba(255,255,255,0.08)",borderRadius:4,color:"#e2e8f0",fontFamily:F,fontSize:12}}/>
            <div style={{fontSize:10,color:"#475569",marginTop:2}}>Min rule fires (absolute count)</div>
          </div>

          {/* min lift */}
          <div>
            <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6}}>
              Min lift: <span style={{color:"#e2e8f0"}}>{(minLift*100).toFixed(1)} pp</span>
            </div>
            <input type="range" min="0" max="0.30" step="0.005" value={minLift}
              onChange={e => setMinLift(parseFloat(e.target.value))} disabled={ip}
              style={{width:"100%"}}/>
            <div style={{fontSize:10,color:"#475569",marginTop:2}}>Precision must beat base rate by this</div>
          </div>
        </div>

        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          <Btn onClick={startMining} disabled={ip} color="#8b5cf6"
            style={{padding:"8px 16px",fontSize:12,fontWeight:700}}>
            {ip ? "Mining..." : "⛏ Mine Rules"}
          </Btn>
          <div style={{fontSize:11,color:"#64748b"}}>
            Takes 5-15 min depending on data volume & horizons
          </div>
        </div>
      </Box>

      {/* ── PROGRESS ── */}
      <Box>
        <Lbl>Mining Progress</Lbl>
        {!progress ? (
          <div style={{fontSize:12,color:"#475569"}}>Loading...</div>
        ) : ip ? (
          <div>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${progress.pct||0}%`,height:"100%",background:"#8b5cf6",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#8b5cf6",fontWeight:600,minWidth:36,textAlign:"right"}}>{progress.pct||0}%</span>
            </div>
            <div style={{fontSize:12,color:"#94a3b8"}}>{progress.message}</div>
          </div>
        ) : phase === "done" ? (
          <div style={{fontSize:12,color:"#22c55e"}}>✓ {progress.message}</div>
        ) : phase === "error" ? (
          <div style={{fontSize:12,color:"#ef4444"}}>✗ {progress.message}</div>
        ) : (
          <div style={{fontSize:12,color:"#475569"}}>Idle</div>
        )}
      </Box>

      {error && (
        <div style={{padding:"8px 12px",borderRadius:6,background:"rgba(239,68,68,0.1)",border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:12}}>
          {error}
        </div>
      )}

      {/* ── CATALOGS ── */}
      <Box>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
          <Lbl>Catalogs ({catalogs.length})</Lbl>
          {catalogs.length > 0 && (
            <Btn onClick={downloadDiagnostic} color="#f97316" style={{padding:"4px 10px",fontSize:11}}>
              ⬇ Download all (rules diagnostic)
            </Btn>
          )}
        </div>
        {catalogs.length === 0 ? (
          <div style={{fontSize:12,color:"#475569",padding:"20px 0",textAlign:"center"}}>
            No catalogs yet. Click "Mine Rules" above to start.
          </div>
        ) : (
          <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
            {catalogs.map(c => {
              const sel = selectedCatalog &&
                selectedCatalog.threshold_bps === c.threshold_bps &&
                selectedCatalog.horizon_hours === c.horizon_hours;
              return (
                <div key={`${c.threshold_bps}_${c.horizon_hours}`}
                  style={{display:"flex",alignItems:"center",gap:4,
                    background:sel?"rgba(139,92,246,0.15)":"rgba(255,255,255,0.04)",
                    border:`1px solid ${sel?"rgba(139,92,246,0.4)":"rgba(255,255,255,0.08)"}`,
                    borderRadius:4,padding:"6px 10px",fontSize:11}}>
                  <span onClick={() => setSelectedCatalog({threshold_bps:c.threshold_bps,horizon_hours:c.horizon_hours})}
                    style={{cursor:"pointer",color:sel?"#e2e8f0":"#94a3b8"}}>
                    <span style={{color:"#8b5cf6",fontWeight:600}}>+{(c.threshold_pct*100).toFixed(1)}%</span>
                    {" / "}<span style={{color:"#e2e8f0"}}>{c.horizon_hours}h</span>
                    {" "}<span style={{color:"#64748b"}}>— {c.rule_count} rules</span>
                    {" "}<span style={{color:"#475569",fontSize:10}}>base {(c.base_rate_test*100).toFixed(1)}%</span>
                  </span>
                  <span onClick={() => deleteCatalog(c.threshold_bps, c.horizon_hours)}
                    style={{cursor:"pointer",color:"#64748b",marginLeft:4,fontSize:14,fontWeight:700}}
                    title="Delete catalog">×</span>
                </div>
              );
            })}
          </div>
        )}
      </Box>

      {/* ── RULES TABLE ── */}
      {selectedCatalog && (
        <Box>
          {!catalogDetail ? (
            <div style={{color:"#475569",fontSize:12}}>Loading catalog...</div>
          ) : catalogDetail.error ? (
            <div style={{color:"#ef4444",fontSize:12}}>{catalogDetail.error}</div>
          ) : (
            <div>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                <Lbl>
                  Rules — +{(catalogDetail.threshold_pct*100).toFixed(1)}% / {catalogDetail.horizon_hours}h
                  {" "}<span style={{color:"#475569",fontWeight:400}}>
                    ({sortedRules.length} rules, base rate test {(catalogDetail.base_rates.test*100).toFixed(1)}%)
                  </span>
                </Lbl>
                <div style={{fontSize:10,color:"#475569"}}>
                  Raw by method: {Object.entries(catalogDetail.stats?.raw_rules_by_method || {}).map(([m,n]) => `${m}:${n}`).join(" · ")}
                </div>
              </div>

              {/* Batch-pin controls */}
              {sortedRules.length > 0 && (
                <div style={{display:"flex",alignItems:"center",gap:8,padding:"10px 12px",marginBottom:10,
                  background:"rgba(34,197,94,0.06)",border:"1px solid rgba(34,197,94,0.15)",borderRadius:4,flexWrap:"wrap"}}>
                  <span style={{fontSize:11,color:"#94a3b8"}}>Batch pin to Live:</span>
                  <Btn onClick={async () => {
                    const n = sortedRules.length;
                    const withDq = sortedRules.filter(r => (r.disqualifiers||[]).length > 0).length;
                    if (!confirm(`Pin ALL ${n} rules (unrefined) + ${withDq} refined variants = up to ${n + withDq} pins.\n\nMax data collection mode (no dedup).\n\nContinue?`)) return;
                    try {
                      const res = await fetch('/api/v2/live/pin_batch', {
                        method: 'POST', headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({
                          threshold_bps: selectedCatalog.threshold_bps,
                          horizon_hours: selectedCatalog.horizon_hours,
                          top_n: 1000,
                          include_disqualifier: true,
                          dedup: false,
                        }),
                      });
                      const d = await res.json();
                      if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
                      alert(`Pinned ${d.n_pinned} rules (${d.n_rules_attempted} rules attempted, ${d.skipped.length} skipped without DQ, ${d.errors.length} errors).\n\nSwitch to Live tab to see them.`);
                    } catch (e) { alert(`Batch pin failed: ${e.message}`); }
                  }} color="#22c55e" style={{padding:"4px 10px",fontSize:10,fontWeight:700}}>
                    📌 Pin ALL ({sortedRules.length} rules + their DQs)
                  </Btn>
                  <Btn onClick={async () => {
                    if (!confirm(`Pin top 5 DEDUPED rules + their top disqualifier variant?\n\n(Near-duplicate rules are collapsed first. You'll typically get 3-10 pins instead of 50.)`)) return;
                    try {
                      const res = await fetch('/api/v2/live/pin_batch', {
                        method: 'POST', headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({
                          threshold_bps: selectedCatalog.threshold_bps,
                          horizon_hours: selectedCatalog.horizon_hours,
                          top_n: 5,
                          include_disqualifier: true,
                          dedup: true,
                        }),
                      });
                      const d = await res.json();
                      if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
                      alert(`Pinned ${d.n_pinned} rules (deduped).`);
                    } catch (e) { alert(`Batch pin failed: ${e.message}`); }
                  }} color="#3b82f6" style={{padding:"4px 10px",fontSize:10}}>
                    📌 Pin top 5 deduped + DQs
                  </Btn>
                  <span style={{flex:1}}/>
                  <Btn onClick={async () => {
                    if (!confirm(`Unpin ALL pinned rules? (This keeps fire/outcome history, just clears the active list.)`)) return;
                    try {
                      const res = await fetch('/api/v2/live/unpin_all', {method: 'POST'});
                      const d = await res.json();
                      alert(`Unpinned ${d.count} rules. History preserved.`);
                    } catch (e) { alert(e.message); }
                  }} color="#ef4444" style={{padding:"4px 10px",fontSize:10}}>
                    🗑 Unpin all
                  </Btn>
                </div>
              )}

              {sortedRules.length === 0 ? (
                <div style={{fontSize:12,color:"#475569",padding:"20px 0",textAlign:"center"}}>
                  No rules in this catalog.{" "}
                  <span style={{color:"#64748b"}}>
                    Try lowering min_precision, min_support, or min_lift and re-mining.
                  </span>
                </div>
              ) : (
                <>
                  <div style={{overflowX:"auto"}}>
                    <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                      <thead>
                        <tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                          <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Methods</th>
                          <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Conditions</th>
                          <th onClick={() => clickHeader("train_precision")} style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                            Train P {sortKey==="train_precision"?(sortDir==="desc"?"↓":"↑"):""}
                          </th>
                          <th onClick={() => clickHeader("test_precision")} style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                            Test P {sortKey==="test_precision"?(sortDir==="desc"?"↓":"↑"):""}
                          </th>
                          <th onClick={() => clickHeader("test_support")} style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                            Test N {sortKey==="test_support"?(sortDir==="desc"?"↓":"↑"):""}
                          </th>
                          <th onClick={() => clickHeader("lift")} style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                            Lift {sortKey==="lift"?(sortDir==="desc"?"↓":"↑"):""}
                          </th>
                          <th onClick={() => clickHeader("gap")} style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                            Gap {sortKey==="gap"?(sortDir==="desc"?"↓":"↑"):""}
                          </th>
                          <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Cross-H</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pageRules.map(r => {
                          const sel = selectedRuleId === r.id;
                          return (
                            <tr key={r.id}
                              onClick={() => setSelectedRuleId(r.id === selectedRuleId ? null : r.id)}
                              style={{borderBottom:"1px solid rgba(255,255,255,0.03)",cursor:"pointer",
                                background:sel?"rgba(139,92,246,0.08)":"transparent"}}>
                              <td style={{padding:"6px 8px",fontSize:10}}>
                                {(r.methods||[]).map(m => (
                                  <span key={m} style={{
                                    display:"inline-block",padding:"1px 6px",marginRight:3,
                                    background:m==="tree"?"#1e3a8a":m==="apriori"?"#4c1d95":"#065f46",
                                    color:"#e2e8f0",borderRadius:3,fontSize:9}}>{m}</span>
                                ))}
                              </td>
                              <td style={{padding:"6px 8px",color:"#94a3b8",fontSize:10,maxWidth:320,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                                {condSummary(r)}
                              </td>
                              <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                                {r.train?.precision != null ? `${(r.train.precision*100).toFixed(1)}%` : "—"}
                              </td>
                              <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(r.test?.precision),fontWeight:700}}>
                                {r.test?.precision != null ? `${(r.test.precision*100).toFixed(1)}%` : "—"}
                              </td>
                              <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                                {r.test?.support ?? 0}
                              </td>
                              <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",
                                color:(r.test?.lift_vs_base ?? 0) >= 0.10 ? "#22c55e" : (r.test?.lift_vs_base ?? 0) >= 0.05 ? "#eab308" : "#ef4444"}}>
                                {r.test?.lift_vs_base != null ? `${(r.test.lift_vs_base*100).toFixed(1)}pp` : "—"}
                              </td>
                              <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:r.overfit_flag?"#ef4444":"#64748b"}}>
                                {r.train_test_gap != null ? `${(r.train_test_gap*100).toFixed(1)}pp${r.overfit_flag?" ⚠":""}` : "—"}
                              </td>
                              <td style={{padding:"6px 8px",fontSize:10}}>
                                {Object.entries(r.cross_horizon || {}).map(([hKey, xh]) => {
                                  const col = xh.lift_vs_base >= 0.05 ? "#22c55e" :
                                              xh.lift_vs_base >= 0 ? "#eab308" : "#ef4444";
                                  return (
                                    <span key={hKey} style={{display:"inline-block",marginRight:6,color:col,fontVariantNumeric:"tabular-nums"}}
                                      title={`${hKey}: P=${(xh.precision*100).toFixed(1)}% lift=${(xh.lift_vs_base*100).toFixed(1)}pp n=${xh.support}`}>
                                      {hKey.replace("h_","")}h:{(xh.precision*100).toFixed(0)}%
                                    </span>);
                                })}
                              </td>
                            </tr>);
                        })}
                      </tbody>
                    </table>
                  </div>

                  {totalPages > 1 && (
                    <div style={{display:"flex",justifyContent:"center",alignItems:"center",gap:6,marginTop:10}}>
                      <Btn onClick={() => setPage(Math.max(0, page-1))} disabled={page===0} style={{padding:"3px 10px",fontSize:11}}>← Prev</Btn>
                      <span style={{fontSize:11,color:"#94a3b8"}}>
                        Page {page+1} of {totalPages} ({pageStart+1}–{pageEnd} of {sortedRules.length})
                      </span>
                      <Btn onClick={() => setPage(Math.min(totalPages-1, page+1))} disabled={page>=totalPages-1} style={{padding:"3px 10px",fontSize:11}}>Next →</Btn>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </Box>
      )}

      {/* ── RULE DETAIL ── */}
      {selectedRuleId && selectedCatalog && (
        <Box>
          {!ruleDetail ? (
            <div style={{color:"#475569",fontSize:12}}>Loading rule detail...</div>
          ) : ruleDetail.error ? (
            <div style={{color:"#ef4444",fontSize:12}}>{ruleDetail.error}</div>
          ) : (
            (() => {
              const r = ruleDetail.rule;
              return (
                <div>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:12}}>
                    <div>
                      <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:4}}>Rule {r.id}</div>
                      <div style={{fontSize:13,color:"#e2e8f0",lineHeight:1.5,fontFamily:F,maxWidth:900}}>
                        {r.english}
                      </div>
                    </div>
                    <div style={{display:"flex",gap:4,alignItems:"center",flexWrap:"wrap"}}>
                      {(r.methods||[]).map(m => (
                        <span key={m} style={{
                          padding:"2px 8px",
                          background:m==="tree"?"#1e3a8a":m==="apriori"?"#4c1d95":"#065f46",
                          color:"#e2e8f0",borderRadius:3,fontSize:10,fontWeight:600}}>{m}</span>
                      ))}
                      <Btn
                        onClick={async () => {
                          try {
                            const res = await fetch('/api/v2/live/pin', {
                              method:'POST',
                              headers:{'Content-Type':'application/json'},
                              body: JSON.stringify({
                                threshold_bps: selectedCatalog.threshold_bps,
                                horizon_hours: selectedCatalog.horizon_hours,
                                rule_id: r.id,
                              }),
                            });
                            const d = await res.json();
                            if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
                            alert(`Pinned to Live (pin_id: ${d.pin.pin_id}).\nGo to the Live tab to see it.`);
                          } catch (e) { alert(`Pin failed: ${e.message}`); }
                        }}
                        color="#22c55e" style={{padding:"3px 10px",fontSize:10,fontWeight:700,marginLeft:6}}>
                        📌 Pin to Live
                      </Btn>
                      {(r.disqualifiers || []).slice(0, 1).map((dq, i) => (
                        <Btn key={i}
                          onClick={async () => {
                            try {
                              // The "direction" stored on the disqualifier tells us which way
                              // to exclude. The condition string uses "<=" or ">=" semantics.
                              const res = await fetch('/api/v2/live/pin', {
                                method:'POST',
                                headers:{'Content-Type':'application/json'},
                                body: JSON.stringify({
                                  threshold_bps: selectedCatalog.threshold_bps,
                                  horizon_hours: selectedCatalog.horizon_hours,
                                  rule_id: r.id,
                                  disqualifier: {
                                    feature: dq.feature,
                                    condition: dq.condition,
                                    thresh: dq.thresh,
                                    direction: dq.direction,
                                  },
                                }),
                              });
                              const d = await res.json();
                              if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
                              alert(`Pinned with disqualifier (pin_id: ${d.pin.pin_id}).\nSee the Live tab.`);
                            } catch (e) { alert(`Pin failed: ${e.message}`); }
                          }}
                          color="#f97316" style={{padding:"3px 10px",fontSize:10,fontWeight:600}}>
                          📌 Pin + top DQ
                        </Btn>
                      ))}
                    </div>
                  </div>

                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
                    {/* Left: metrics across splits */}
                    <div>
                      <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:8}}>Metrics</div>
                      <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                        <thead>
                          <tr style={{borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
                            <th style={{padding:"4px 6px",textAlign:"left",color:"#64748b",fontSize:10}}>Split</th>
                            <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Precision</th>
                            <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Support</th>
                            <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Lift</th>
                            <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Recall</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[["train",r.train],["val",r.val],["test",r.test]].map(([k,m]) => m && (
                            <tr key={k} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                              <td style={{padding:"4px 6px",color:"#94a3b8",textTransform:"uppercase",fontSize:10}}>{k}</td>
                              <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(m.precision),fontWeight:600}}>
                                {(m.precision*100).toFixed(1)}%
                              </td>
                              <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>{m.support}</td>
                              <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:m.lift_vs_base>=0.05?"#22c55e":m.lift_vs_base>=0?"#eab308":"#ef4444"}}>
                                {(m.lift_vs_base*100).toFixed(1)}pp
                              </td>
                              <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                                {(m.recall*100).toFixed(1)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <div style={{fontSize:10,color:"#64748b",marginTop:6,lineHeight:1.5}}>
                        Train-test gap: <span style={{color:r.overfit_flag?"#ef4444":"#94a3b8",fontVariantNumeric:"tabular-nums",fontWeight:600}}>
                          {(r.train_test_gap*100).toFixed(1)}pp
                        </span>
                        {r.overfit_flag && <span style={{color:"#ef4444",marginLeft:6}}>⚠ flagged as likely overfit (gap &gt; 15pp)</span>}
                      </div>

                      <div style={{marginTop:16}}>
                        <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:8}}>Cross-horizon retest (test set)</div>
                        {Object.keys(r.cross_horizon || {}).length === 0 ? (
                          <div style={{fontSize:11,color:"#475569"}}>No other horizons were mined in this run.</div>
                        ) : (
                          <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                            <thead>
                              <tr style={{borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
                                <th style={{padding:"4px 6px",textAlign:"left",color:"#64748b",fontSize:10}}>Horizon</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Precision</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Base</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Lift</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>N</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(r.cross_horizon).map(([hKey,xh]) => (
                                <tr key={hKey} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                                  <td style={{padding:"4px 6px",color:"#94a3b8"}}>{hKey.replace("h_","")}h</td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(xh.precision),fontWeight:600}}>
                                    {(xh.precision*100).toFixed(1)}%
                                  </td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#475569"}}>
                                    {(xh.base_rate*100).toFixed(1)}%
                                  </td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:xh.lift_vs_base>=0.05?"#22c55e":xh.lift_vs_base>=0?"#eab308":"#ef4444"}}>
                                    {(xh.lift_vs_base*100).toFixed(1)}pp
                                  </td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>{xh.support}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}
                      </div>
                    </div>

                    {/* Right: full conditions + disqualifiers */}
                    <div>
                      <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:8}}>Conditions (full)</div>
                      <div style={{background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:4,padding:10}}>
                        {(r.conditions||[]).map((c, i) => {
                          const binLabels = ruleDetail.bin_labels?.[c.feature];
                          const binStrs = c.bins.map(b =>
                            binLabels && binLabels[b] ? binLabels[b] : `bin${b}`);
                          return (
                            <div key={i} style={{fontSize:11,color:"#e2e8f0",marginBottom:4,fontFamily:F}}>
                              {i > 0 && <span style={{color:"#64748b"}}>AND </span>}
                              <span style={{color:"#8b5cf6"}}>{c.feature}</span>
                              <span style={{color:"#64748b"}}> in </span>
                              <span>{"{ " + binStrs.join(", ") + " }"}</span>
                            </div>
                          );
                        })}
                      </div>

                      <div style={{marginTop:16}}>
                        <div style={{fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:8}}>Disqualifiers (training FPs → TPs)</div>
                        {!r.disqualifiers || r.disqualifiers.length === 0 ? (
                          <div style={{fontSize:11,color:"#475569"}}>No disqualifier candidates found.{" "}
                          <span style={{color:"#64748b"}}>(Need ≥5 TPs and ≥5 FPs on training fires.)</span>
                          </div>
                        ) : (
                          <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                            <thead>
                              <tr style={{borderBottom:"1px solid rgba(255,255,255,0.06)"}}>
                                <th style={{padding:"4px 6px",textAlign:"left",color:"#64748b",fontSize:10}}>Exclude if</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Drops FPs</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Drops TPs</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>New P</th>
                                <th style={{padding:"4px 6px",textAlign:"right",color:"#64748b",fontSize:10}}>Gain</th>
                              </tr>
                            </thead>
                            <tbody>
                              {r.disqualifiers.slice(0,5).map((d, i) => (
                                <tr key={i} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                                  <td style={{padding:"4px 6px",color:"#94a3b8",fontFamily:F,fontSize:10}}>{d.condition}</td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#ef4444"}}>{d.excluded_fp}</td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>{d.excluded_tp}</td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(d.new_precision_train),fontWeight:600}}>
                                    {(d.new_precision_train*100).toFixed(1)}%
                                  </td>
                                  <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#22c55e"}}>
                                    +{(d.precision_gain_train*100).toFixed(1)}pp
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}
                        <div style={{fontSize:10,color:"#64748b",marginTop:6,lineHeight:1.5}}>
                          Disqualifiers are computed on <b>training</b> false positives. Add to rule only after retesting on val/test.
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()
          )}
        </Box>
      )}
    </div>
  );
}

// ─── LIVE (v2 Stage 3) ───────────────────────────────────────────
// Live scanner + outcome recording dashboard.
//
// This tab is the empirical test: pinned rules are evaluated against fresh
// Coinbase data every 4 hours (matching our scan slot cadence), fires are
// persisted, and after each rule's horizon elapses, an outcome is recorded.
// The "live precision" per rule is then compared to the validation precision
// from the catalog — if they match, the rule is real; if live collapses to
// base rate, we learn the in-sample validation was measuring noise.
//
// Four subsections:
//   1. Pinned rules overview — validation precision, live precision, delta
//   2. Recent fires — what fired most recently
//   3. Scan & resolve controls — manual triggers for scan/outcome jobs
//   4. Diagnostic download

function LiveTab() {
  const [pinned, setPinned] = useState([]);
  const [stats, setStats] = useState(null);
  const [scanStatus, setScanStatus] = useState(null);
  const [fires, setFires] = useState([]);
  const [selectedPin, setSelectedPin] = useState(null);   // pin_id filter for fires
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [backtestTestP, setBacktestTestP] = useState({}); // pin_id -> test precision from most recent backtest

  // Load the most recent backtest report's test precisions (for comparison column).
  // Done once on mount + whenever pinned rules change.
  useEffect(() => {
    fetch('/api/v2/live/backtest/list').then(r => r.json()).then(async d => {
      const list = d.backtests || [];
      if (list.length === 0) return;
      // list is sorted newest first
      const newest = list[0];
      const r2 = await fetch(`/api/v2/live/backtest/${newest.backtest_id}`);
      const report = await r2.json();
      const map = {};
      for (const rr of (report.per_rule || [])) {
        const tp = (rr.by_split?.test || {}).precision;
        if (tp != null) map[rr.pin_id] = tp;
      }
      setBacktestTestP(map);
    }).catch(() => {});
  }, [pinned.length]);

  // Poll every 10s for everything; 3s for scan status while active
  useEffect(() => {
    const fetchAll = () => {
      fetch('/api/v2/live/pinned').then(r => r.json())
        .then(d => setPinned(d.rules || [])).catch(() => {});
      fetch('/api/v2/live/stats').then(r => r.json()).then(setStats).catch(() => {});
      fetch('/api/v2/live/scan/status').then(r => r.json()).then(setScanStatus).catch(() => {});
      const q = selectedPin ? `?pin_id=${encodeURIComponent(selectedPin)}&limit=50` : '?limit=50';
      fetch(`/api/v2/live/fires${q}`).then(r => r.json())
        .then(d => setFires(d.fires || [])).catch(() => {});
    };
    fetchAll();
    const iv = setInterval(fetchAll, scanStatus?.inProgress ? 3000 : 10000);
    return () => clearInterval(iv);
  }, [selectedPin, scanStatus?.inProgress]);

  const runScanNow = useCallback(async () => {
    setBusy(true); setError(null);
    try {
      const r = await fetch('/api/v2/live/scan', {method: 'POST'});
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
      if (d.status === 'already_in_progress') setError("Scan already in progress");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }, []);

  const resolveOutcomesNow = useCallback(async () => {
    setBusy(true); setError(null);
    try {
      const r = await fetch('/api/v2/live/record_outcomes', {method: 'POST'});
      if (!r.ok) { const d = await r.json(); throw new Error(d.error); }
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }, []);

  const unpinRule = useCallback(async (pin_id) => {
    if (!confirm(`Unpin ${pin_id}? Existing fires remain recorded.`)) return;
    try {
      const r = await fetch(`/api/v2/live/pin/${encodeURIComponent(pin_id)}`, {method: 'DELETE'});
      if (!r.ok) { const d = await r.json(); throw new Error(d.error); }
      // Refresh
      const rr = await fetch('/api/v2/live/pinned'); const dd = await rr.json();
      setPinned(dd.rules || []);
      if (selectedPin === pin_id) setSelectedPin(null);
    } catch (e) { alert(e.message); }
  }, [selectedPin]);

  const downloadDiag = useCallback(async () => {
    try {
      const r = await fetch('/api/v2/live/diagnostic');
      const b = await r.blob();
      const fn = r.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1] || 'live_diag.json';
      const u = URL.createObjectURL(b);
      const a = document.createElement('a');
      a.href = u; a.download = fn;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(u);
    } catch (e) { alert(e.message); }
  }, []);

  const precColor = (p) => {
    if (p == null) return "#64748b";
    if (p >= 0.70) return "#22c55e";
    if (p >= 0.55) return "#eab308";
    return "#ef4444";
  };

  const deltaColor = (d) => {
    if (d == null) return "#64748b";
    if (d >= -0.05) return "#22c55e";   // within 5pp of validation = holding up
    if (d >= -0.15) return "#eab308";   // 5-15pp below = warning
    return "#ef4444";                   // >15pp below = signal likely collapsed
  };

  const byRule = stats?.by_rule || {};
  const totals = stats?.totals || {};

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>

      {/* ── BANNER ── */}
      <Box>
        <div style={{fontSize:11,color:"#22c55e",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6,fontWeight:700}}>
          v2 Stage 3 — Live Scanner & Outcome Recording
        </div>
        <div style={{fontSize:12,color:"#94a3b8",lineHeight:1.6}}>
          Pinned rules are evaluated against fresh Coinbase data every 4 hours (at :06 after
          each 4h slot). Every fire is recorded. After each rule's horizon elapses, the outcome
          is resolved automatically (was +X% hit?). Compare live precision to validation
          precision — if they match, the pattern is real.
          Pin rules from the <span style={{color:"#8b5cf6"}}>Rules</span> tab.
        </div>
      </Box>

      {/* ── CONTROLS ── */}
      <Box>
        <div style={{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
          <Btn onClick={runScanNow} disabled={busy||scanStatus?.inProgress} color="#22c55e"
            style={{padding:"6px 12px",fontSize:11,fontWeight:700}}>
            {scanStatus?.inProgress ? "Scanning..." : "▶ Run scan now"}
          </Btn>
          <Btn onClick={resolveOutcomesNow} disabled={busy} color="#f97316"
            style={{padding:"6px 12px",fontSize:11}}>
            ↻ Resolve outcomes now
          </Btn>
          {pinned.length > 0 && (
            <Btn onClick={downloadDiag} color="#f97316"
              style={{padding:"6px 12px",fontSize:11}}>
              ⬇ Download live diagnostic
            </Btn>
          )}
          <div style={{flex:1}}/>
          {scanStatus?.lastResult && !scanStatus.inProgress && (
            <div style={{fontSize:11,color:"#64748b"}}>
              Last scan: {scanStatus.lastResult.status === "ok"
                ? `${scanStatus.lastResult.n_fires} fires across ${scanStatus.lastResult.n_coins_evaluated} coins in ${scanStatus.lastResult.elapsed_sec}s`
                : `status: ${scanStatus.lastResult.status}`}
            </div>
          )}
        </div>
        {scanStatus?.inProgress && (
          <div style={{marginTop:8,fontSize:11,color:"#22c55e"}}>
            ⚡ Scan in progress — this polls every 3s until complete.
          </div>
        )}
        {error && (
          <div style={{marginTop:8,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.1)",
            border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:11}}>
            {error}
          </div>
        )}
      </Box>

      {/* ── PINNED RULES STATS ── */}
      <Box>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
          <Lbl>Pinned Rules ({pinned.length})</Lbl>
          {totals.n_fires_resolved > 0 && (
            <div style={{fontSize:11,color:"#94a3b8"}}>
              Total: {totals.n_fires_resolved} resolved / {totals.n_fires_pending} pending /{" "}
              <span style={{color:precColor(totals.overall_live_precision),fontWeight:600}}>
                {totals.overall_live_precision != null ? `${(totals.overall_live_precision*100).toFixed(1)}%` : "—"}
              </span> overall
            </div>
          )}
        </div>

        {/* Prune-to-winners quick action — only show if there are >3 pins */}
        {pinned.length > 3 && (
          <div style={{padding:"8px 10px",marginBottom:10,
            background:"rgba(245,158,11,0.06)",border:"1px solid rgba(245,158,11,0.2)",
            borderRadius:4,display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
            <span style={{fontSize:11,color:"#f59e0b"}}>⚡ Quick action:</span>
            <span style={{fontSize:11,color:"#94a3b8"}}>
              Based on backtest analysis, keep only the 2 rules that generalize on test.
            </span>
            <span style={{flex:1}}/>
            <Btn onClick={async () => {
              const keepIds = [
                "11c305c85a8e",                                       // pattern 1 unrefined, 2-condition (simplest), Test P 65.6%
                "7447e71f2ab7_dq_resistance_dist_atr_exclude_if_greater", // pattern 2 refined w/ best DQ threshold, Test P 61.6%
              ];
              if (!confirm(`Keep only these 2 pinned rules (the ones that generalize on test):\n\n`
                + `  • bb_width + rsi_14 (Test P 65.6%, Train→Test gap -0.5pp, 343 test fires)\n`
                + `  • bb_width + support_dist_atr +DQ(resistance ≤ 2.125) (Test P 61.6%, 125 test fires)\n\n`
                + `Unpin everything else? Fire/outcome history is preserved.`)) return;
              try {
                const r = await fetch('/api/v2/live/unpin_except', {
                  method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({keep_pin_ids: keepIds}),
                });
                const d = await r.json();
                if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
                alert(`Kept: ${d.kept}\nUnpinned: ${d.unpinned}\n\nKept pin_ids:\n  • ${d.kept_pin_ids.join('\n  • ')}`);
              } catch (e) { alert(e.message); }
            }} color="#f59e0b" style={{padding:"4px 10px",fontSize:10,fontWeight:700}}>
              🎯 Keep only the 2 generalizing rules
            </Btn>
          </div>
        )}

        {/* Pin capitulation-bounce (from brute-force mining, OOS test +4.83%) */}
        <div style={{display:"flex",alignItems:"center",gap:8,padding:"6px 10px",marginBottom:10,
          background:"rgba(168,85,247,0.06)",border:"1px solid rgba(168,85,247,0.15)",borderRadius:4}}>
          <span style={{fontSize:11,color:"#a855f7"}}>⚡ Quick action:</span>
          <span style={{fontSize:11,color:"#94a3b8"}}>
            Pin the capitulation-bounce rule (6h drop ≥ 10% → +2% in 12h).
            Test precision 57.8%, OOS +4.83% on 65 trades.
          </span>
          <span style={{flex:1}}/>
          <Btn onClick={async () => {
            if (!confirm(`Pin the capitulation-bounce rule?\n\n`
              + `Rule: ret_6h_pct < -10% → +2% within 12h\n`
              + `Test precision: 57.8% (vs 20.6% base rate)\n`
              + `Test EV: +4.83% over 87 days (65 trades), annualized +21.9%\n\n`
              + `Note: Val slice showed -2.34% during Oct 2025 crash period.\n`
              + `This rule has real regime risk; pinning lets us collect forward data.`)) return;
            try {
              const r = await fetch('/api/v2/live/pin_custom', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                  pin_id: "capitulation_bounce_6h_12h",
                  english: "ret_6h_pct < -10% → +2% within 12h",
                  conditions: [
                    {"feature": "ret_6h_pct", "op": "lt", "threshold": -10.0},
                  ],
                  threshold_pct: 0.02,
                  horizon_hours: 12,
                  validation_precision: 0.578,
                  validation_support: 65,
                  validation_base_rate: 0.206,
                  notes: "From brute-force mining on 360d (train/val/test temporal split). Test EV +4.83%, annualized +21.9%. Val -2.34% during crash regime. Mined 2026-04-23.",
                }),
              });
              const d = await r.json();
              if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
              alert(`Pinned.\n\npin_id: ${d.pin_id}\n\nThe live scanner will now evaluate this rule on every scan and record outcomes.`);
            } catch (e) { alert(e.message); }
          }} color="#a855f7" style={{padding:"4px 10px",fontSize:10,fontWeight:700}}>
            📌 Pin capitulation-bounce rule
          </Btn>
        </div>

        {/* Pin FILTERED variant — adds BTC 3d > -5% regime filter */}
        <div style={{display:"flex",alignItems:"center",gap:8,padding:"6px 10px",marginBottom:10,
          background:"rgba(236,72,153,0.06)",border:"1px solid rgba(236,72,153,0.15)",borderRadius:4}}>
          <span style={{fontSize:11,color:"#ec4899"}}>⚡ Quick action:</span>
          <span style={{fontSize:11,color:"#94a3b8"}}>
            Pin the FILTERED variant (adds BTC 3d return &gt; -5% regime filter).
            Survived all 3 splits positive; ~+10% annualized expectation.
          </span>
          <span style={{flex:1}}/>
          <Btn onClick={async () => {
            if (!confirm(`Pin the FILTERED capitulation-bounce rule?\n\n`
              + `Rule: ret_6h_pct < -10% AND btc_3d_ret_pct > -5% → +2% within 12h\n`
              + `Rationale: skip setups during BTC-wide crashes.\n\n`
              + `Filtered performance (intra-path sim, MAKER fees 65bps):\n`
              + `  Train (201d, 422 trades): +2.83%, max DD -6.8%\n`
              + `  Val   ( 73d, 175 trades): +4.53%, max DD -2.9%\n`
              + `  Test  ( 73d, 129 trades): +2.08%, max DD -2.0%\n\n`
              + `Combined ~12 months: ~+9.9%, annualized ~+10%.\n\n`
              + `Pinning this alongside the raw rule lets us compare live firerates & outcomes.`)) return;
            try {
              const r = await fetch('/api/v2/live/pin_custom', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                  pin_id: "capitulation_bounce_filtered",
                  english: "ret_6h_pct < -10% AND btc_3d_ret_pct > -5% → +2% within 12h",
                  conditions: [
                    {"feature": "ret_6h_pct", "op": "lt", "threshold": -10.0},
                    {"feature": "btc_3d_ret_pct", "op": "gt", "threshold": -5.0},
                  ],
                  threshold_pct: 0.02,
                  horizon_hours: 12,
                  validation_precision: 0.55,
                  validation_support: 726,
                  validation_base_rate: 0.206,
                  notes: "Capitulation-bounce with BTC regime filter (3d > -5). Only accepted filter out of 45 tested, passed acceptance criteria across all 3 splits. Train n=422 (+2.83%, max DD -6.8%), val n=175 (+4.53%, max DD -2.9%), test n=129 (+2.08%, max DD -2.0%). Annualized ~+10% combined. Mined 2026-04-23.",
                }),
              });
              const d = await r.json();
              if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
              alert(`Pinned FILTERED rule.\n\npin_id: ${d.pin_id}\n\nBoth the raw and filtered rules are now live. Forward fires will be recorded separately so we can compare.`);
            } catch (e) { alert(e.message); }
          }} color="#ec4899" style={{padding:"4px 10px",fontSize:10,fontWeight:700}}>
            📌 Pin filtered variant (+ BTC regime)
          </Btn>
        </div>

        {pinned.length === 0 ? (
          <div style={{fontSize:12,color:"#475569",padding:"20px 0",textAlign:"center"}}>
            No pinned rules. Go to the <span style={{color:"#8b5cf6",fontWeight:600}}>Rules tab</span>,
            click a rule, and use the "Pin to Live" button in its detail view.
          </div>
        ) : (
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
              <thead>
                <tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                  <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Rule</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Threshold/H</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}} title="Precision from mining's held-out test slice (stored with the rule)">Catalog P</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}} title="Test-slice precision from most recent backtest run">Backtest P</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Live P</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}} title="Live P minus Backtest P (if available) else Live P minus Catalog P">Δ</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Hits/Res</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Pending</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Avg Max%</th>
                  <th style={{padding:"6px 8px",textAlign:"center",color:"#64748b",fontSize:10,fontWeight:500}}></th>
                </tr>
              </thead>
              <tbody>
                {pinned.map(p => {
                  const s = byRule[p.pin_id] || {};
                  const isSel = selectedPin === p.pin_id;
                  const btP = backtestTestP[p.pin_id];
                  // Prefer backtest P as the baseline if available, else catalog P
                  const deltaBaseline = btP != null ? btP : p.validation_precision;
                  const deltaVal = (s.live_precision != null && deltaBaseline != null)
                    ? s.live_precision - deltaBaseline : null;
                  return (
                    <tr key={p.pin_id}
                      onClick={() => setSelectedPin(isSel ? null : p.pin_id)}
                      style={{borderBottom:"1px solid rgba(255,255,255,0.03)",cursor:"pointer",
                        background:isSel?"rgba(34,197,94,0.08)":"transparent"}}>
                      <td style={{padding:"6px 8px",maxWidth:380}}>
                        <div style={{fontSize:11,color:"#e2e8f0",fontFamily:F,
                          whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>
                          {p.english}
                          {p.disqualifier && (
                            <span style={{color:"#f97316",marginLeft:6,fontSize:10}}>
                              (+ DQ: {p.disqualifier.condition})
                            </span>
                          )}
                        </div>
                        <div style={{fontSize:9,color:"#475569",fontFamily:F}}>
                          pin_id: {p.pin_id.slice(0, 28)}{p.pin_id.length > 28 ? "..." : ""}
                        </div>
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                        +{(p.threshold_pct*100).toFixed(1)}% / {p.horizon_hours}h
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(p.validation_precision),fontWeight:600}}>
                        {p.validation_precision != null ? `${(p.validation_precision*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(backtestTestP[p.pin_id]),fontWeight:600}}>
                        {backtestTestP[p.pin_id] != null ? `${(backtestTestP[p.pin_id]*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:precColor(s.live_precision),fontWeight:700}}>
                        {s.live_precision != null ? `${(s.live_precision*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:deltaColor(deltaVal)}}>
                        {deltaVal != null ? `${(deltaVal*100).toFixed(1)}pp` : "—"}
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                        {s.n_hits ?? 0}/{s.n_fires_resolved ?? 0}
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#64748b"}}>
                        {s.n_fires_pending ?? 0}
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                        {s.avg_max_pct != null ? `${(s.avg_max_pct*100).toFixed(2)}%` : "—"}
                      </td>
                      <td style={{padding:"6px 8px",textAlign:"center"}}>
                        <span onClick={(e) => { e.stopPropagation(); unpinRule(p.pin_id); }}
                          style={{cursor:"pointer",color:"#64748b",fontSize:16,fontWeight:700}}
                          title="Unpin rule">×</span>
                      </td>
                    </tr>);
                })}
              </tbody>
            </table>
          </div>
        )}

        {pinned.length > 0 && (
          <div style={{fontSize:10,color:"#475569",marginTop:10,lineHeight:1.6}}>
            Δ compares Live P to <b>Backtest P</b> if backtest has run, else Catalog P.{" "}
            <span style={{color:"#22c55e"}}>Green</span>: within 5pp of baseline (healthy).{" "}
            <span style={{color:"#eab308"}}>Yellow</span>: 5-15pp below (warning).{" "}
            <span style={{color:"#ef4444"}}>Red</span>: &gt;15pp below (signal likely collapsed).{" "}
            Click a row to filter fires by that rule.
          </div>
        )}
      </Box>

      {/* ── RECENT FIRES ── */}
      {pinned.length > 0 && (
        <Box>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
            <Lbl>Recent Fires {selectedPin ? `(filtered to ${selectedPin.slice(0,20)}...)` : "(all rules)"}</Lbl>
            {selectedPin && (
              <Btn onClick={() => setSelectedPin(null)} style={{padding:"3px 8px",fontSize:10}}>
                Clear filter
              </Btn>
            )}
          </div>
          {fires.length === 0 ? (
            <div style={{fontSize:12,color:"#475569",padding:"20px 0",textAlign:"center"}}>
              No fires yet{selectedPin?" for this rule":""}. Fires appear here after the next
              scheduled scan (every 4h at :06) or when you click "Run scan now".
            </div>
          ) : (
            <div style={{overflowX:"auto",maxHeight:400,overflowY:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                <thead style={{position:"sticky",top:0,background:"#0c0f14"}}>
                  <tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                    <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Scan Time</th>
                    <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Product</th>
                    <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Entry</th>
                    <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Rule</th>
                    <th style={{padding:"6px 8px",textAlign:"center",color:"#64748b",fontSize:10,fontWeight:500}}>Status</th>
                    <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Max %</th>
                    <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Final %</th>
                  </tr>
                </thead>
                <tbody>
                  {fires.slice().reverse().map(f => {
                    const oc = f.outcome || {};
                    const statusColor = !f.resolved ? "#64748b"
                      : oc.error ? "#ef4444"
                      : oc.hit ? "#22c55e" : "#ef4444";
                    const statusText = !f.resolved ? "Pending"
                      : oc.error ? `Err`
                      : oc.hit ? "✓ Hit" : "✗ Miss";
                    return (
                      <tr key={f.fire_id} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                        <td style={{padding:"4px 6px",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                          {f.scan_time ? new Date(f.scan_time).toLocaleString() : "—"}
                        </td>
                        <td style={{padding:"4px 6px",fontSize:10,color:"#e2e8f0",fontWeight:600}}>
                          {f.product}
                        </td>
                        <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                          {f.entry_price != null ? `$${f.entry_price.toLocaleString(undefined,{maximumFractionDigits:4})}` : "—"}
                        </td>
                        <td style={{padding:"4px 6px",fontSize:9,color:"#64748b",fontFamily:F}}>
                          {f.pin_id?.slice(0, 18)}{f.pin_id?.length > 18 ? "…" : ""}
                          {f.disqualifier_applied && <span style={{color:"#f97316",marginLeft:4}}>[+DQ]</span>}
                        </td>
                        <td style={{padding:"4px 6px",textAlign:"center",fontSize:10,fontWeight:700,color:statusColor}}>
                          {statusText}
                        </td>
                        <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                          {oc.max_pct != null ? `${(oc.max_pct*100).toFixed(2)}%` : "—"}
                        </td>
                        <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                          {oc.final_pct != null ? `${(oc.final_pct*100).toFixed(2)}%` : "—"}
                        </td>
                      </tr>);
                  })}
                </tbody>
              </table>
            </div>
          )}
          <div style={{fontSize:10,color:"#475569",marginTop:8,lineHeight:1.5}}>
            Showing last {fires.length} fires. Max % = highest price reached during horizon.
            Final % = close at end of horizon. Hit means max % ≥ threshold within horizon.
          </div>
        </Box>
      )}

      {/* ── BACKTEST ── */}
      <BacktestPanel pinned={pinned} />

      {/* ── PAPER TRADING SIMULATION ── */}
      <PaperSimPanel pinned={pinned} />
    </div>
  );
}

// ─── BACKTEST PANEL ──────────────────────────────────────────────
// Evaluates pinned rules against historical cached bars. Unlike live, this
// uses rows that were ALREADY used for training/validation — so precision
// numbers are inflated. We split results 3 ways (train/val/test) so you can
// see train (inflated), val (slightly out-of-sample), test (true held-out).
// The TEST-slice precision should reproduce the catalog's test precision
// within ~1pp — that's the reproducibility check.

function BacktestPanel({ pinned }) {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [btStatus, setBtStatus] = useState({inProgress: false, progress: {pct:0, msg:"idle"}});
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [sortKey, setSortKey] = useState("overall_precision");
  const [error, setError] = useState(null);

  // Poll status whenever a backtest is running; also load the history list
  useEffect(() => {
    const loadStatus = () => {
      fetch('/api/v2/live/backtest/status').then(r => r.json())
        .then(setBtStatus).catch(() => {});
      fetch('/api/v2/live/backtest/list').then(r => r.json())
        .then(d => setHistory(d.backtests || [])).catch(() => {});
    };
    loadStatus();
    const iv = setInterval(loadStatus, btStatus.inProgress ? 2000 : 15000);
    return () => clearInterval(iv);
  }, [btStatus.inProgress]);

  // When a backtest completes and there's no selected report, auto-select newest
  useEffect(() => {
    if (!selected && history.length > 0 && !btStatus.inProgress) {
      setSelected(history[0].backtest_id);
    }
  }, [history, btStatus.inProgress]);

  // Load the selected backtest report
  useEffect(() => {
    if (!selected) { setSelectedReport(null); return; }
    fetch(`/api/v2/live/backtest/${selected}`).then(r => r.json())
      .then(setSelectedReport).catch(() => {});
  }, [selected]);

  const runBacktest = async () => {
    setError(null);
    const body = {};
    if (startDate) body.start_date = startDate;
    if (endDate) body.end_date = endDate;
    try {
      const r = await fetch('/api/v2/live/backtest', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
      if (d.status === 'already_in_progress') setError("Backtest already in progress");
    } catch (e) { setError(e.message); }
  };

  const precColor = (p) => {
    if (p == null) return "#64748b";
    if (p >= 0.70) return "#22c55e";
    if (p >= 0.55) return "#eab308";
    return "#ef4444";
  };

  const sortedRules = useMemo(() => {
    if (!selectedReport?.per_rule) return [];
    const rows = selectedReport.per_rule.slice();
    const accessor = {
      overall_precision: r => r.overall?.precision ?? -1,
      overall_fires: r => r.overall?.n_fires ?? -1,
      test_precision: r => r.by_split?.test?.precision ?? -1,
      val_precision: r => r.by_split?.val?.precision ?? -1,
      train_precision: r => r.by_split?.train?.precision ?? -1,
      validation_precision: r => r.validation_precision ?? -1,
    }[sortKey] || (r => -1);
    rows.sort((a, b) => accessor(b) - accessor(a));
    return rows;
  }, [selectedReport, sortKey]);

  return (
    <Box>
      <div style={{marginBottom:12}}>
        <Lbl>🔁 Backtest Pinned Rules Against Historical Data</Lbl>
        <div style={{fontSize:11,color:"#94a3b8",marginTop:6,lineHeight:1.6}}>
          Evaluates all pinned rules against cached bars (full 180-day training window by default).
          Results include <b>train</b> slice (inflated — rules were mined on this data),
          <b>val</b> slice (slightly out-of-sample), and <b>test</b> slice (truly held out).
          The <b>test</b> precision should roughly reproduce the catalog number.
          Runs as a background job; expect ~30 sec for feature build + ~2 sec per rule.
        </div>
      </div>

      {/* Controls */}
      <div style={{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap",marginBottom:10,padding:"8px 10px",background:"rgba(168,85,247,0.06)",border:"1px solid rgba(168,85,247,0.15)",borderRadius:4}}>
        <span style={{fontSize:11,color:"#94a3b8"}}>Date range:</span>
        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
          style={{padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
            border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
        <span style={{fontSize:10,color:"#64748b"}}>to</span>
        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
          style={{padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
            border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
        <span style={{fontSize:10,color:"#64748b"}}>
          (leave blank = full cached window)
        </span>
        <span style={{flex:1}}/>
        <Btn onClick={runBacktest} disabled={btStatus.inProgress || pinned.length === 0}
          color="#a855f7" style={{padding:"5px 12px",fontSize:11,fontWeight:700}}>
          {btStatus.inProgress ? `Running ${btStatus.progress?.pct ?? 0}%...` : "▶ Run Backtest"}
        </Btn>
      </div>

      {btStatus.inProgress && (
        <div style={{marginBottom:10}}>
          <div style={{background:"rgba(168,85,247,0.15)",borderRadius:3,height:6,overflow:"hidden"}}>
            <div style={{width:`${btStatus.progress?.pct ?? 0}%`,height:6,background:"#a855f7",transition:"width 0.3s"}}/>
          </div>
          <div style={{fontSize:10,color:"#a855f7",marginTop:4}}>
            {btStatus.progress?.msg ?? "..."}
          </div>
        </div>
      )}

      {error && (
        <div style={{marginBottom:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.1)",
          border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:11}}>
          {error}
        </div>
      )}

      {/* History selector */}
      {history.length > 0 && (
        <div style={{display:"flex",gap:6,alignItems:"center",marginBottom:10,flexWrap:"wrap"}}>
          <span style={{fontSize:10,color:"#64748b"}}>Past runs:</span>
          {history.slice(0, 5).map(h => (
            <span key={h.backtest_id}
              onClick={() => setSelected(h.backtest_id)}
              style={{padding:"3px 8px",fontSize:10,cursor:"pointer",borderRadius:3,
                background: selected === h.backtest_id ? "rgba(168,85,247,0.25)" : "rgba(255,255,255,0.05)",
                border: selected === h.backtest_id ? "1px solid #a855f7" : "1px solid rgba(255,255,255,0.08)",
                color: selected === h.backtest_id ? "#e9d5ff" : "#94a3b8"}}>
              {new Date(h.generated_at).toLocaleString()} ({h.n_rules_evaluated} rules)
            </span>
          ))}
        </div>
      )}

      {/* Results */}
      {selectedReport && selectedReport.per_rule && (
        <div>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:8,flexWrap:"wrap"}}>
            <div style={{fontSize:10,color:"#64748b"}}>
              Backtest <b>{selectedReport.backtest_id}</b> — range {selectedReport.date_range?.actual_first} to {selectedReport.date_range?.actual_last}
              {" "}({selectedReport.date_range?.n_rows} rows, {selectedReport.date_range?.n_distinct_dates} dates)
            </div>
            <span style={{flex:1}}/>
            <Btn onClick={() => {
              const blob = new Blob([JSON.stringify(selectedReport, null, 2)], {type: "application/json"});
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `${selectedReport.backtest_id}.json`;
              document.body.appendChild(a); a.click();
              document.body.removeChild(a); URL.revokeObjectURL(url);
            }} color="#a855f7" style={{padding:"3px 8px",fontSize:10}}>
              ⬇ Download report
            </Btn>
          </div>
          <div style={{overflowX:"auto",maxHeight:600,overflowY:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
              <thead style={{position:"sticky",top:0,background:"#0c0f14"}}>
                <tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                  <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Rule</th>
                  <th onClick={() => setSortKey("overall_fires")} style={{padding:"6px 8px",textAlign:"right",color:sortKey==="overall_fires"?"#a855f7":"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                    Fires {sortKey==="overall_fires"?"↓":""}
                  </th>
                  <th onClick={() => setSortKey("overall_precision")} style={{padding:"6px 8px",textAlign:"right",color:sortKey==="overall_precision"?"#a855f7":"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                    Overall P {sortKey==="overall_precision"?"↓":""}
                  </th>
                  <th onClick={() => setSortKey("train_precision")} style={{padding:"6px 8px",textAlign:"right",color:sortKey==="train_precision"?"#a855f7":"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                    Train P {sortKey==="train_precision"?"↓":""}
                  </th>
                  <th onClick={() => setSortKey("val_precision")} style={{padding:"6px 8px",textAlign:"right",color:sortKey==="val_precision"?"#a855f7":"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                    Val P {sortKey==="val_precision"?"↓":""}
                  </th>
                  <th onClick={() => setSortKey("test_precision")} style={{padding:"6px 8px",textAlign:"right",color:sortKey==="test_precision"?"#a855f7":"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                    Test P {sortKey==="test_precision"?"↓":""}
                  </th>
                  <th onClick={() => setSortKey("validation_precision")} style={{padding:"6px 8px",textAlign:"right",color:sortKey==="validation_precision"?"#a855f7":"#64748b",fontSize:10,fontWeight:500,cursor:"pointer"}}>
                    Catalog P {sortKey==="validation_precision"?"↓":""}
                  </th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Coins</th>
                </tr>
              </thead>
              <tbody>
                {sortedRules.map(r => {
                  const ov = r.overall || {};
                  const tr = r.by_split?.train || {};
                  const va = r.by_split?.val || {};
                  const te = r.by_split?.test || {};
                  // Reproducibility check: is test precision close to catalog's validation?
                  const reproDelta = (te.precision != null && r.validation_precision != null)
                    ? te.precision - r.validation_precision : null;
                  const hasDq = r.disqualifier != null;
                  return (
                    <tr key={r.pin_id} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                      <td style={{padding:"4px 6px",maxWidth:300}}>
                        <div style={{fontSize:10,color:"#e2e8f0",fontFamily:F,
                          whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>
                          {r.english}
                          {hasDq && <span style={{color:"#f97316",marginLeft:4}}>[+DQ]</span>}
                        </div>
                        <div style={{fontSize:9,color:"#475569",fontFamily:F}}>
                          {r.pin_id.slice(0, 30)}{r.pin_id.length > 30 ? "..." : ""}
                        </div>
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                        {ov.n_hits ?? 0}/{ov.n_fires ?? 0}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,fontWeight:700,color:precColor(ov.precision),fontVariantNumeric:"tabular-nums"}}>
                        {ov.precision != null ? `${(ov.precision*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:precColor(tr.precision),fontVariantNumeric:"tabular-nums"}}>
                        {tr.precision != null ? `${(tr.precision*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:precColor(va.precision),fontVariantNumeric:"tabular-nums"}}>
                        {va.precision != null ? `${(va.precision*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,fontWeight:700,color:precColor(te.precision),fontVariantNumeric:"tabular-nums"}}
                        title={reproDelta != null ? `Δ vs catalog: ${(reproDelta*100).toFixed(1)}pp` : ""}>
                        {te.precision != null ? `${(te.precision*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                        {r.validation_precision != null ? `${(r.validation_precision*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                        {ov.n_distinct_coins ?? 0}
                      </td>
                    </tr>);
                })}
              </tbody>
            </table>
          </div>
          <div style={{fontSize:10,color:"#475569",marginTop:10,lineHeight:1.6}}>
            <b style={{color:"#94a3b8"}}>Reading this table:</b>{" "}
            <b>Train P</b> uses rows the rules were mined on — expect inflation.{" "}
            <b>Test P</b> uses held-out rows — should match <b>Catalog P</b> within ~1pp (reproducibility check).{" "}
            <b>Val P</b> is an intermediate out-of-sample measurement.{" "}
            If Test P ≫ Catalog P, we have a reproducibility bug (fire detection differs from mining-time logic).{" "}
            If Test P &lt; Catalog P materially, same concern.
          </div>
        </div>
      )}

      {!selectedReport && !btStatus.inProgress && history.length === 0 && (
        <div style={{fontSize:12,color:"#475569",padding:"20px 0",textAlign:"center"}}>
          No backtests run yet. Click <b>Run Backtest</b> above to start one.
        </div>
      )}
    </Box>
  );
}

// ─── PAPER SIMULATION PANEL ──────────────────────────────────────
// Replays rule fires with realistic TP/SL exits + trading costs.
// This is the "does this actually make money after frictions" check.

function PaperSimPanel({ pinned }) {
  const [executionMode, setExecutionMode] = useState("taker_market");   // or "maker_limit"
  const [tpPct, setTpPct] = useState(2.0);
  const [slPct, setSlPct] = useState(2.0);
  // Taker-mode fee fields
  const [costBps, setCostBps] = useState(30);
  const [slipBps, setSlipBps] = useState(10);
  const [useNextOpen, setUseNextOpen] = useState(true);
  // Maker-mode fee fields
  const [makerFeeBps, setMakerFeeBps] = useState(25);
  const [takerFeeBps, setTakerFeeBps] = useState(40);
  const [makerSlipBps, setMakerSlipBps] = useState(2);
  const [takerSlipBps, setTakerSlipBps] = useState(15);
  const [posPct, setPosPct] = useState(5.0);
  const [capital, setCapital] = useState(10000);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [status, setStatus] = useState({inProgress: false, progress: {pct:0, msg:"idle"}});
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = () => {
      fetch('/api/v2/live/paper_sim/status').then(r => r.json())
        .then(setStatus).catch(() => {});
      fetch('/api/v2/live/paper_sim/list').then(r => r.json())
        .then(d => setHistory(d.sims || [])).catch(() => {});
    };
    load();
    const iv = setInterval(load, status.inProgress ? 2000 : 15000);
    return () => clearInterval(iv);
  }, [status.inProgress]);

  useEffect(() => {
    if (!selected && history.length > 0 && !status.inProgress) {
      setSelected(history[0].sim_id);
    }
  }, [history, status.inProgress]);

  useEffect(() => {
    if (!selected) { setSelectedReport(null); return; }
    fetch(`/api/v2/live/paper_sim/${selected}`).then(r => r.json())
      .then(setSelectedReport).catch(() => {});
  }, [selected]);

  const runSim = async () => {
    setError(null);
    const body = {
      tp_pct: tpPct / 100, sl_pct: slPct / 100,
      position_size_pct: posPct / 100, starting_capital: capital,
      execution_mode: executionMode,
    };
    if (executionMode === "maker_limit") {
      body.maker_fee_bps = makerFeeBps;
      body.taker_fee_bps = takerFeeBps;
      body.maker_slippage_bps = makerSlipBps;
      body.taker_slippage_bps = takerSlipBps;
    } else {
      body.cost_bps_per_side = costBps;
      body.slippage_bps_per_side = slipBps;
      body.use_next_bar_open = useNextOpen;
    }
    if (startDate) body.start_date = startDate;
    if (endDate) body.end_date = endDate;
    try {
      const r = await fetch('/api/v2/live/paper_sim', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
      if (d.status === 'already_in_progress') setError("Simulation already in progress");
    } catch (e) { setError(e.message); }
  };

  const pnlColor = (pct) => pct == null ? "#64748b"
    : pct > 0.005 ? "#22c55e" : pct < -0.005 ? "#ef4444" : "#eab308";

  const pf = selectedReport?.portfolio || {};
  const rules = selectedReport?.per_rule || [];

  return (
    <Box>
      <div style={{marginBottom:12}}>
        <Lbl>💵 Paper Trading Simulation</Lbl>
        <div style={{fontSize:11,color:"#94a3b8",marginTop:6,lineHeight:1.6}}>
          Replays every rule fire as a paper trade with realistic exits (TP/SL) and trading costs.
          This is the honest "does this make money" check — precision alone isn't enough.
          Choose between <b>taker</b> (market orders, models execution lag) or <b>maker</b> (limit orders,
          lower fees but some fires won't fill).
        </div>
      </div>

      {/* Execution mode selector */}
      <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:8,padding:"8px 10px",
        background:"rgba(59,130,246,0.06)",border:"1px solid rgba(59,130,246,0.2)",borderRadius:4}}>
        <span style={{fontSize:11,color:"#93c5fd",fontWeight:600}}>Execution mode:</span>
        <label style={{fontSize:11,color:"#e2e8f0",cursor:"pointer"}}>
          <input type="radio" name="exec_mode" checked={executionMode === "taker_market"}
            onChange={() => setExecutionMode("taker_market")}
            style={{marginRight:4}}/>
          Taker (market orders)
        </label>
        <label style={{fontSize:11,color:"#e2e8f0",cursor:"pointer"}}>
          <input type="radio" name="exec_mode" checked={executionMode === "maker_limit"}
            onChange={() => setExecutionMode("maker_limit")}
            style={{marginRight:4}}/>
          Maker (limit orders; limits may not fill)
        </label>
        <span style={{flex:1}}/>
        <span style={{fontSize:9,color:"#64748b"}}>
          {executionMode === "maker_limit"
            ? "Entry: limit at scan close (filled if price dips). Exit: TP=limit (maker), SL/horizon=market (taker)."
            : "Entry: market at next bar open (models execution lag). All exits: market."}
        </span>
      </div>

      {/* Config */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(150px,1fr))",
        gap:8,padding:"10px",marginBottom:10,background:"rgba(34,197,94,0.06)",
        border:"1px solid rgba(34,197,94,0.15)",borderRadius:4}}>
        <div>
          <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>TP %</div>
          <input type="number" step="0.1" value={tpPct} onChange={e => setTpPct(parseFloat(e.target.value) || 0)}
            style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
              border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
        </div>
        <div>
          <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>SL %</div>
          <input type="number" step="0.1" value={slPct} onChange={e => setSlPct(parseFloat(e.target.value) || 0)}
            style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
              border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
        </div>
        {executionMode === "taker_market" && (<>
          <div>
            <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Cost bps per side</div>
            <input type="number" value={costBps} onChange={e => setCostBps(parseFloat(e.target.value) || 0)}
              style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
                border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
          </div>
          <div>
            <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Slippage bps</div>
            <input type="number" value={slipBps} onChange={e => setSlipBps(parseFloat(e.target.value) || 0)}
              style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
                border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
          </div>
        </>)}
        {executionMode === "maker_limit" && (<>
          <div>
            <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Maker fee bps</div>
            <input type="number" value={makerFeeBps} onChange={e => setMakerFeeBps(parseFloat(e.target.value) || 0)}
              style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
                border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
          </div>
          <div>
            <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Taker fee bps (SL exits)</div>
            <input type="number" value={takerFeeBps} onChange={e => setTakerFeeBps(parseFloat(e.target.value) || 0)}
              style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
                border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
          </div>
          <div>
            <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Maker slip bps</div>
            <input type="number" value={makerSlipBps} onChange={e => setMakerSlipBps(parseFloat(e.target.value) || 0)}
              style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
                border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
          </div>
          <div>
            <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Taker slip bps</div>
            <input type="number" value={takerSlipBps} onChange={e => setTakerSlipBps(parseFloat(e.target.value) || 0)}
              style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
                border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
          </div>
        </>)}
        <div>
          <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Position size %</div>
          <input type="number" step="0.5" value={posPct} onChange={e => setPosPct(parseFloat(e.target.value) || 0)}
            style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
              border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
        </div>
        <div>
          <div style={{fontSize:9,color:"#64748b",marginBottom:3}}>Starting capital $</div>
          <input type="number" value={capital} onChange={e => setCapital(parseFloat(e.target.value) || 0)}
            style={{width:"100%",padding:"4px 6px",fontSize:11,background:"#0c0f14",color:"#e2e8f0",
              border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}} />
        </div>
        {executionMode === "taker_market" && (
          <div style={{display:"flex",alignItems:"flex-end"}}>
            <label style={{fontSize:10,color:"#94a3b8",cursor:"pointer",userSelect:"none"}}>
              <input type="checkbox" checked={useNextOpen} onChange={e => setUseNextOpen(e.target.checked)}
                style={{marginRight:6}}/>
              Entry at next-bar open (model exec lag)
            </label>
          </div>
        )}
      </div>

      <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:10,flexWrap:"wrap"}}>
        <span style={{fontSize:10,color:"#64748b"}}>Date range (optional):</span>
        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
          style={{padding:"3px 6px",fontSize:10,background:"#0c0f14",color:"#e2e8f0",
            border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}}/>
        <span style={{fontSize:9,color:"#64748b"}}>to</span>
        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
          style={{padding:"3px 6px",fontSize:10,background:"#0c0f14",color:"#e2e8f0",
            border:"1px solid rgba(255,255,255,0.1)",borderRadius:3}}/>
        <span style={{flex:1}}/>
        <Btn onClick={runSim} disabled={status.inProgress || pinned.length === 0}
          color="#22c55e" style={{padding:"5px 14px",fontSize:11,fontWeight:700}}>
          {status.inProgress ? `Running ${status.progress?.pct ?? 0}%...` : "💵 Run Paper Simulation"}
        </Btn>
      </div>

      {status.inProgress && (
        <div style={{marginBottom:10}}>
          <div style={{background:"rgba(34,197,94,0.15)",borderRadius:3,height:6,overflow:"hidden"}}>
            <div style={{width:`${status.progress?.pct ?? 0}%`,height:6,background:"#22c55e",transition:"width 0.3s"}}/>
          </div>
          <div style={{fontSize:10,color:"#22c55e",marginTop:4}}>{status.progress?.msg ?? "..."}</div>
        </div>
      )}

      {error && (
        <div style={{marginBottom:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.1)",
          border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:11}}>{error}</div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div style={{display:"flex",gap:6,alignItems:"center",marginBottom:10,flexWrap:"wrap"}}>
          <span style={{fontSize:10,color:"#64748b"}}>Past runs:</span>
          {history.slice(0, 5).map(h => (
            <span key={h.sim_id} onClick={() => setSelected(h.sim_id)}
              style={{padding:"3px 8px",fontSize:10,cursor:"pointer",borderRadius:3,
                background: selected === h.sim_id ? "rgba(34,197,94,0.25)" : "rgba(255,255,255,0.05)",
                border: selected === h.sim_id ? "1px solid #22c55e" : "1px solid rgba(255,255,255,0.08)",
                color: selected === h.sim_id ? "#d1fae5" : "#94a3b8"}}>
              {new Date(h.generated_at).toLocaleString()} — TP{(h.config?.tp_pct*100 || 2).toFixed(1)}%/
              SL{(h.config?.sl_pct*100 || 2).toFixed(1)}% — {h.portfolio?.total_return_pct?.toFixed(1) ?? "?"}%
            </span>
          ))}
        </div>
      )}

      {/* Portfolio summary */}
      {selectedReport && (
        <>
          <div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:14,padding:"10px",
            background:"rgba(168,85,247,0.06)",border:"1px solid rgba(168,85,247,0.15)",borderRadius:4}}>
            <div style={{flex:"1 1 120px"}}>
              <div style={{fontSize:9,color:"#64748b",textTransform:"uppercase"}}>Start → End</div>
              <div style={{fontSize:12,color:"#e2e8f0",fontWeight:600,fontVariantNumeric:"tabular-nums"}}>
                ${pf.starting_capital?.toLocaleString()} → ${pf.final_equity?.toLocaleString()}
              </div>
            </div>
            <div style={{flex:"1 1 120px"}}>
              <div style={{fontSize:9,color:"#64748b",textTransform:"uppercase"}}>Total Return</div>
              <div style={{fontSize:14,fontWeight:700,color:pnlColor(pf.total_return_pct/100),fontVariantNumeric:"tabular-nums"}}>
                {pf.total_return_pct?.toFixed(2) ?? "—"}%
              </div>
            </div>
            <div style={{flex:"1 1 120px"}}>
              <div style={{fontSize:9,color:"#64748b",textTransform:"uppercase"}}>Max Drawdown</div>
              <div style={{fontSize:12,color:"#ef4444",fontWeight:600,fontVariantNumeric:"tabular-nums"}}>
                {pf.max_drawdown_pct?.toFixed(2) ?? "—"}%
              </div>
            </div>
            <div style={{flex:"1 1 100px"}}>
              <div style={{fontSize:9,color:"#64748b",textTransform:"uppercase"}}>Trades</div>
              <div style={{fontSize:12,color:"#e2e8f0",fontWeight:600,fontVariantNumeric:"tabular-nums"}}>
                {pf.n_trades?.toLocaleString() ?? "—"}
              </div>
            </div>
            <div style={{flex:"1 1 100px"}}>
              <div style={{fontSize:9,color:"#64748b",textTransform:"uppercase"}}>Pos size</div>
              <div style={{fontSize:12,color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>
                {(pf.position_size_pct*100)?.toFixed(1) ?? "—"}%
              </div>
            </div>
            <Btn onClick={() => {
              const blob = new Blob([JSON.stringify(selectedReport, null, 2)], {type:"application/json"});
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a"); a.href = url; a.download = `${selectedReport.sim_id}.json`;
              document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
            }} color="#a855f7" style={{padding:"3px 8px",fontSize:10}}>⬇ Download</Btn>
          </div>

          {/* Per-rule breakdown */}
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
              <thead>
                <tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                  <th style={{padding:"6px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500}}>Rule</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}} title="For maker mode: filled / (filled + no-fill)">Trades</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}} title="Mean P&L per trade before costs">Raw/trade</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}} title="Mean P&L per trade after costs">Net/trade</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Win rate</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Total net</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Test net</th>
                  <th style={{padding:"6px 8px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>Exits</th>
                </tr>
              </thead>
              <tbody>
                {rules.sort((a,b) => (b.aggregate?.total_net_pct || 0) - (a.aggregate?.total_net_pct || 0)).map(r => {
                  const agg = r.aggregate || {};
                  const ts = agg.by_split?.test || {};
                  const isMaker = (selectedReport?.config?.execution_mode === "maker_limit");
                  return (
                    <tr key={r.pin_id} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                      <td style={{padding:"4px 6px",maxWidth:330}}>
                        <div style={{fontSize:11,color:"#e2e8f0",fontFamily:F,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>
                          {r.english}
                          {r.disqualifier && <span style={{color:"#f97316",marginLeft:4,fontSize:10}}>[+DQ]</span>}
                        </div>
                        <div style={{fontSize:9,color:"#475569",fontFamily:F}}>{r.pin_id.slice(0,30)}</div>
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                        {isMaker && agg.n_no_fill > 0
                          ? `${agg.n_trades ?? 0}/${(agg.n_trades ?? 0) + (agg.n_no_fill ?? 0)}`
                          : (agg.n_trades ?? 0)}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:pnlColor(agg.mean_raw_pnl_pct)}}>
                        {agg.mean_raw_pnl_pct != null ? `${(agg.mean_raw_pnl_pct*100).toFixed(2)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",fontWeight:700,color:pnlColor(agg.mean_net_pnl_pct)}}>
                        {agg.mean_net_pnl_pct != null ? `${(agg.mean_net_pnl_pct*100).toFixed(2)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:"#94a3b8"}}>
                        {agg.win_rate_net != null ? `${(agg.win_rate_net*100).toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",fontWeight:700,color:pnlColor(agg.total_net_pct/100)}}>
                        {agg.total_net_pct != null ? `${agg.total_net_pct.toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontVariantNumeric:"tabular-nums",color:pnlColor(ts.total_net_pct/100)}}>
                        {ts.total_net_pct != null ? `${ts.total_net_pct.toFixed(1)}%` : "—"}
                      </td>
                      <td style={{padding:"4px 6px",textAlign:"right",fontSize:9,color:"#64748b",fontFamily:F}}>
                        {agg.exit_types ? Object.entries(agg.exit_types).map(([k,v]) => `${k[0]}${v}`).join("/") : "—"}
                      </td>
                    </tr>);
                })}
              </tbody>
            </table>
          </div>
          <div style={{fontSize:10,color:"#475569",marginTop:10,lineHeight:1.6}}>
            <b>Raw/trade</b>: mean P&L per trade before costs.{" "}
            <b>Net/trade</b>: after {(selectedReport.config?.cost_bps_per_side + selectedReport.config?.slippage_bps_per_side)*2 / 100}% round-trip costs.{" "}
            <b>Total net</b>: sum of all trade P&Ls (not compounded, not capital-adjusted).{" "}
            <b>Test net</b>: same but restricted to the test-split rows (most honest number).{" "}
            <b>Exits</b>: t=TP, s=SL, h=horizon.
          </div>
        </>
      )}

      {!selectedReport && !status.inProgress && history.length === 0 && (
        <div style={{fontSize:12,color:"#475569",padding:"20px 0",textAlign:"center"}}>
          No simulations run yet. Adjust config above and click <b>Run Paper Simulation</b>.
        </div>
      )}
    </Box>
  );
}

// ─── DATA EXPORT TAB ─────────────────────────────────────────────
// Downloads analytical CSV presets for offline Claude analysis.
// Each preset is a focused slice: daily overview, hourly features,
// event windows, time-of-day, or rolling correlations.

function DataExportTab() {
  const [presets, setPresets] = useState([]);
  const [downloading, setDownloading] = useState(null);   // e.g. "hourly_features:current"
  const [error, setError] = useState(null);
  const [priorStatus, setPriorStatus] = useState(null);

  useEffect(() => {
    fetch('/api/data_export/presets').then(r => r.json())
      .then(d => setPresets(d.presets || []))
      .catch(e => setError(e.message));
  }, []);

  // Poll prior-fetch status
  useEffect(() => {
    const poll = () => {
      fetch('/api/data_export/fetch_prior/status').then(r => r.json())
        .then(setPriorStatus).catch(() => {});
    };
    poll();
    const iv = setInterval(poll, priorStatus?.inProgress ? 3000 : 20000);
    return () => clearInterval(iv);
  }, [priorStatus?.inProgress]);

  const startPriorFetch = async () => {
    setError(null);
    try {
      const r = await fetch('/api/data_export/fetch_prior', { method: 'POST' });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
      if (d.status === 'cache_fresh') {
        setError(`Prior-180d cache already exists (${d.age_hours}h old). Ready to use.`);
      }
    } catch (e) { setError(e.message); }
  };

  const downloadPreset = async (name, window = "current") => {
    const key = `${name}:${window}`;
    setDownloading(key); setError(null);
    try {
      const url = window === "prior"
        ? `/api/data_export/${name}?window=prior`
        : `/api/data_export/${name}`;
      const r = await fetch(url);
      if (!r.ok) {
        const d = await r.json();
        throw new Error(d.error || `HTTP ${r.status}`);
      }
      const blob = await r.blob();
      const fname = r.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1]
        || `coinbase_export_${name}_${window}.zip`;
      const u = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = u; a.download = fname;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(u);
    } catch (e) {
      setError(`${name} (${window}): ${e.message}`);
    } finally {
      setDownloading(null);
    }
  };

  // Map preset → guidance so users know which to upload for which question
  const guidance = {
    daily_overview: {
      startHere: true,
      bestFor: "Broad orientation. Which coins have the highest returns/volatility? Which pairs move together?",
      claudePrompt: `Given daily_returns.csv, universe.csv, daily_correlations.csv from my Coinbase scanner: what stands out? Look for outliers, unexpected correlations, and any patterns worth investigating further with a more targeted preset.`,
    },
    hourly_features: {
      bestFor: "Lead/lag analysis. Does coin A consistently move before coin B at 6h cadence?",
      claudePrompt: `Given hourly_features.csv (6-hour snapshots per coin with returns, RSI, volume_z, vs_BTC): compute cross-correlation of coin returns at different lags to identify lead/lag relationships. Which coins lead BTC? Which ones lag by 1-2 slots?`,
    },
    event_windows: {
      bestFor: "Pre-event signatures. What features are elevated/suppressed in the 6 hours BEFORE a +5% move?",
      claudePrompt: `Given events.csv and event_features.csv (windows around +5% moves): look at the bars with offset_bars < 0 (pre-event) and identify what features (volume, cumulative return, BTC return) are systematically different from a random baseline. Do big moves pre-announce themselves?`,
    },
    time_of_day: {
      bestFor: "Calendar effects. Are certain hours of UTC day systematically bullish/bearish?",
      claudePrompt: `Given hour_of_day_stats.csv and day_of_week_stats.csv: are there statistically meaningful time-of-day or day-of-week effects? Calculate standard errors and identify coins with the most pronounced patterns.`,
    },
    rolling_correlations: {
      bestFor: "Regime analysis. When did coins decouple from BTC? How do correlations shift during high-vol periods?",
      claudePrompt: `Given rolling_corr_btc.csv and volatility_regimes.csv: identify coins whose correlation to BTC varies most dramatically. Are correlations higher during high-vol regimes? Which coins are most independent?`,
    },
  };

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <div style={{fontSize:11,color:"#06b6d4",letterSpacing:0.5,textTransform:"uppercase",marginBottom:6,fontWeight:700}}>
          Data Export — analytical CSV presets
        </div>
        <div style={{fontSize:12,color:"#94a3b8",lineHeight:1.6}}>
          Export pre-analyzed slices of the cached Coinbase market data for offline analysis.
          Each preset downloads as a ZIP with 1-3 CSVs + a self-describing README.
          Upload one to a fresh Claude conversation with the suggested prompt to investigate
          that specific question. Iterate by downloading different presets as findings suggest.
        </div>
      </Box>

      {/* Prior-180d OOS fetch controls */}
      <Box>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:12,flexWrap:"wrap"}}>
          <div>
            <div style={{fontSize:11,color:"#a855f7",letterSpacing:0.5,textTransform:"uppercase",fontWeight:700,marginBottom:4}}>
              Prior 180d (Out-of-Sample) Historical Data
            </div>
            <div style={{fontSize:11,color:"#94a3b8",lineHeight:1.5}}>
              Fetch the 180-day window BEFORE the current operational cache (i.e. 360d ago → 180d ago).
              Use this to test whether patterns found in current data hold on truly out-of-sample history.
              One-time ~30-60 min fetch; then the purple "Prior 180d (OOS)" buttons on each preset become active.
            </div>
            {priorStatus?.cache_exists && (
              <div style={{fontSize:10,color:"#d8b4fe",marginTop:6}}>
                ✓ Prior-180d cache available
                {priorStatus.cache_age_hours != null && ` (built ${priorStatus.cache_age_hours}h ago)`}
              </div>
            )}
            {priorStatus?.inProgress && (
              <div style={{marginTop:8}}>
                <div style={{background:"rgba(168,85,247,0.15)",borderRadius:3,height:6,overflow:"hidden"}}>
                  <div style={{width:`${priorStatus.progress?.pct ?? 0}%`,height:6,background:"#a855f7",transition:"width 0.3s"}}/>
                </div>
                <div style={{fontSize:10,color:"#a855f7",marginTop:4}}>
                  {priorStatus.progress?.msg ?? ""} ({priorStatus.progress?.pct ?? 0}%)
                </div>
              </div>
            )}
          </div>
          <Btn onClick={startPriorFetch}
            disabled={priorStatus?.inProgress}
            color="#a855f7" style={{padding:"6px 14px",fontSize:11,fontWeight:700,whiteSpace:"nowrap"}}>
            {priorStatus?.inProgress
              ? `Fetching ${priorStatus.progress?.pct ?? 0}%...`
              : (priorStatus?.cache_exists ? "🔄 Refetch Prior 180d" : "📥 Fetch Prior 180d")}
          </Btn>
        </div>
      </Box>

      {error && (
        <Box>
          <div style={{padding:"8px 12px",borderRadius:4,background:"rgba(239,68,68,0.1)",
            border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:12}}>
            {error}
          </div>
        </Box>
      )}

      {presets.length === 0 && !error && (
        <Box><div style={{color:"#475569",fontSize:12}}>Loading presets...</div></Box>
      )}

      {presets.map(preset => {
        const g = guidance[preset.name] || {};
        return (
          <Box key={preset.name}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:12,flexWrap:"wrap"}}>
              <div style={{flex:"1 1 400px"}}>
                <div style={{fontSize:13,color:"#e2e8f0",fontWeight:700,marginBottom:4}}>
                  {g.startHere && <span style={{color:"#22c55e",fontSize:10,marginRight:6,
                    padding:"2px 6px",background:"rgba(34,197,94,0.15)",borderRadius:3}}>START HERE</span>}
                  {preset.name}
                </div>
                <div style={{fontSize:11,color:"#94a3b8",marginBottom:6}}>
                  {preset.description}
                </div>
                {g.bestFor && (
                  <div style={{fontSize:11,color:"#cbd5e1",marginBottom:8,lineHeight:1.5}}>
                    <span style={{color:"#64748b",fontSize:10,textTransform:"uppercase"}}>Best for: </span>
                    {g.bestFor}
                  </div>
                )}
                {g.claudePrompt && (
                  <div style={{fontSize:10,color:"#64748b",marginTop:6,marginBottom:2,textTransform:"uppercase",letterSpacing:0.5}}>
                    Suggested prompt for a fresh Claude conversation:
                  </div>
                )}
                {g.claudePrompt && (
                  <div style={{fontSize:10,padding:"6px 8px",background:"rgba(6,182,212,0.06)",
                    border:"1px solid rgba(6,182,212,0.15)",borderRadius:3,
                    color:"#a5f3fc",fontFamily:F,lineHeight:1.5}}>
                    {g.claudePrompt}
                  </div>
                )}
              </div>
              <div style={{display:"flex",flexDirection:"column",gap:4,alignItems:"flex-end"}}>
                <Btn onClick={() => downloadPreset(preset.name, "current")}
                  disabled={downloading !== null}
                  color="#06b6d4" style={{padding:"6px 14px",fontSize:11,fontWeight:700,whiteSpace:"nowrap"}}>
                  {downloading === `${preset.name}:current` ? "Building..." : "⬇ Current 180d"}
                </Btn>
                <Btn onClick={() => downloadPreset(preset.name, "prior")}
                  disabled={downloading !== null || !priorStatus?.cache_exists}
                  color="#a855f7" style={{padding:"4px 10px",fontSize:10,fontWeight:600,whiteSpace:"nowrap",opacity: priorStatus?.cache_exists ? 1 : 0.4}}>
                  {downloading === `${preset.name}:prior` ? "Building..." : "⬇ Prior 180d (OOS)"}
                </Btn>
              </div>
            </div>
          </Box>);
      })}

      <Box>
        <div style={{fontSize:10,color:"#475569",lineHeight:1.6}}>
          <b style={{color:"#94a3b8"}}>Workflow:</b> Open a new Claude conversation, upload the ZIP,
          paste the suggested prompt. If a finding looks promising, come back here and download another
          preset to drill down. Each preset is independent; you don't need to keep any project context.
          <br/><br/>
          <b style={{color:"#94a3b8"}}>Why separate downloads?</b> Each ZIP is sized for Claude's context
          window (~1-10MB). Uploading everything at once slows analysis. One focused upload per question
          gives cleaner results.
        </div>
      </Box>
    </div>
  );
}

// ─── MAIN ────────────────────────────────────────────────────────
export default function CoinbaseScanner() {
  const [scanHour,setScanHour]=useState(12);
  const [tab,setTab]=useState("live");
  const [data,setData]=useState([]);
  const [source,setSource]=useState("loading");
  const [loading,setLoading]=useState(true);
  const [lastUpdate,setLastUpdate]=useState(null);
  const [elapsed,setElapsed]=useState(null);
  const [modelWR10,setModelWR10]=useState(null);
  const [modelPnL10,setModelPnL10]=useState(null);
  const [scanInfo,setScanInfo]=useState(null);
  const [health,setHealth]=useState(null);
  const [error,setError]=useState(null);
  const [message,setMessage]=useState(null);

  useEffect(()=>{fetch('/api/health').then(r=>r.json()).then(setHealth).catch(()=>{});},[]);

  const fetchScan=useCallback(async(hour,force=false)=>{
    setLoading(true);setError(null);setMessage(null);
    try{
      const url=force?`/api/scan/${hour}/refresh`:`/api/scan/${hour}`;
      const r=await fetch(url,force?{method:'POST'}:{});
      if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.error||`HTTP ${r.status}`);}
      const d=await r.json();
      setData(d.data||[]);setSource(d.source||"offline");setLastUpdate(d.timestamp);
      setElapsed(d.elapsed||null);setModelWR10(d.modelWR10||null);setModelPnL10(d.modelPnL10||null);
      setScanInfo({tp_pct:d.tp_pct,sl_pct:d.sl_pct,breakeven:d.breakeven,horizonHours:d.horizonHours});
      setMessage(d.message||null);
    }catch(err){setError(err.message);setSource("error");setData([]);}
    finally{setLoading(false);}
  },[]);

  useEffect(()=>{fetchScan(scanHour);},[scanHour,fetchScan]);
  useEffect(()=>{if(source!=="live")return;const iv=setInterval(()=>fetchScan(scanHour),5*60*1000);return()=>clearInterval(iv);},[source,scanHour,fetchScan]);

  const downloadDiag=useCallback(async()=>{
    try{const r=await fetch('/api/diagnostic');const b=await r.blob();
      const fn=r.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1]||`diag.json`;
      const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=fn;document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(u);
    }catch(e){alert(e.message);}
  },[]);

  const tabs=[{id:"live",l:"Live (v2.3)",c:"#22c55e"},{id:"rules",l:"Rules (v2.2)",c:"#8b5cf6"},{id:"export",l:"Data Export",c:"#06b6d4"},{id:"v2",l:"v2 Classifier"},{id:"scanner",l:"Scanner (v1)"},{id:"training",l:"Training (v1)",c:"#8b5cf6"},{id:"outcomes",l:"Outcomes (v1)",c:"#22c55e"},{id:"status",l:"Status"}];

  return (
    <div style={{fontFamily:F,background:"#0c0f14",color:"#e2e8f0",minHeight:"100vh"}}>
      <div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",padding:"12px 20px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{fontSize:15,fontWeight:800,letterSpacing:1}}>COINBASE SCANNER</div>
          <SourceBadge source={loading?"loading":source} trained={health?.trained}/>
        </div>
        <div style={{display:"flex",gap:4,fontSize:11,flexWrap:"wrap",alignItems:"center"}}>
          <span style={{color:"#eab308",fontWeight:600}}>
            {tab==="live" ? "v2 Stage 3 — Live scanner + outcome recording" :
              tab==="rules" ? "v2 Stage 2 — Rule mining (absolute +%, multi-horizon)" :
              tab==="export" ? "Data Export — analytical CSV presets for offline analysis" :
              tab==="v2" ? "v2 — Vol-normalized threshold classifier (Stage 1)" :
              health ? `TP +${health.tp_pct}% / SL -${health.sl_pct}% / ${health.horizonHours||4}h horizon (BE ${health.breakeven}%)` : "Loading..."}
          </span>
          {lastUpdate&&tab!=="v2"&&tab!=="rules"&&tab!=="live"&&tab!=="export"&&<><span style={{color:"#334155",margin:"0 4px"}}>|</span><span style={{color:"#94a3b8"}}>{new Date(lastUpdate).toLocaleString()}</span></>}
        </div>
      </div>

      <div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",padding:"8px 20px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:10,color:"#475569",textTransform:"uppercase",letterSpacing:0.5,marginRight:4}}>Scan UTC</span>
          {SCAN_HOURS.map(h=><Btn key={h} active={h===scanHour} onClick={()=>setScanHour(h)}>{fmtHour(h)}</Btn>)}
          <Btn onClick={()=>fetchScan(scanHour,true)} disabled={!health?.trained} style={{marginLeft:4}}>↻ Refresh</Btn>
          <Btn onClick={downloadDiag} color="#f97316" style={{marginLeft:4}}>⬇ Diagnostic</Btn>
        </div>
        <div style={{display:"flex",gap:2}}>
          {tabs.map(t=><Btn key={t.id} active={t.id===tab} onClick={()=>setTab(t.id)} color={t.c||"#3b82f6"}>{t.l}</Btn>)}
        </div>
      </div>

      {error&&<div style={{margin:"12px 20px 0",padding:"8px 12px",borderRadius:6,background:"rgba(239,68,68,0.1)",border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:12}}>{error}</div>}

      <div style={{padding:"16px 20px"}}>
        {tab==="live"&&<LiveTab/>}
        {tab==="rules"&&<RulesTab/>}
        {tab==="export"&&<DataExportTab/>}
        {tab==="v2"&&<V2Tab/>}
        {tab==="scanner"&&<ScannerTab data={data} scanHour={scanHour} source={source} elapsed={elapsed} message={message} modelWR10={modelWR10} modelPnL10={modelPnL10} health={health} scanInfo={scanInfo}/>}
        {tab==="training"&&<TrainingTab/>}
        {tab==="outcomes"&&<OutcomesTab/>}
        {tab==="status"&&<StatusTab health={health}/>}
      </div>
    </div>);
}
