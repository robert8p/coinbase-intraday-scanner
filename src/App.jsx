import { useState, useEffect, useCallback } from "react";

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

// ─── MAIN ────────────────────────────────────────────────────────
export default function CoinbaseScanner() {
  const [scanHour,setScanHour]=useState(12);
  const [tab,setTab]=useState("v2");
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

  const tabs=[{id:"v2",l:"v2 Research",c:"#8b5cf6"},{id:"scanner",l:"Scanner (v1)"},{id:"training",l:"Training (v1)",c:"#8b5cf6"},{id:"outcomes",l:"Outcomes (v1)",c:"#22c55e"},{id:"status",l:"Status"}];

  return (
    <div style={{fontFamily:F,background:"#0c0f14",color:"#e2e8f0",minHeight:"100vh"}}>
      <div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",padding:"12px 20px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{fontSize:15,fontWeight:800,letterSpacing:1}}>COINBASE SCANNER</div>
          <SourceBadge source={loading?"loading":source} trained={health?.trained}/>
        </div>
        <div style={{display:"flex",gap:4,fontSize:11,flexWrap:"wrap",alignItems:"center"}}>
          <span style={{color:"#eab308",fontWeight:600}}>
            {tab==="v2" ? "v2 — Vol-normalized threshold classifier (Stage 1)" :
              health ? `TP +${health.tp_pct}% / SL -${health.sl_pct}% / ${health.horizonHours||4}h horizon (BE ${health.breakeven}%)` : "Loading..."}
          </span>
          {lastUpdate&&tab!=="v2"&&<><span style={{color:"#334155",margin:"0 4px"}}>|</span><span style={{color:"#94a3b8"}}>{new Date(lastUpdate).toLocaleString()}</span></>}
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
        {tab==="v2"&&<V2Tab/>}
        {tab==="scanner"&&<ScannerTab data={data} scanHour={scanHour} source={source} elapsed={elapsed} message={message} modelWR10={modelWR10} modelPnL10={modelPnL10} health={health} scanInfo={scanInfo}/>}
        {tab==="training"&&<TrainingTab/>}
        {tab==="outcomes"&&<OutcomesTab/>}
        {tab==="status"&&<StatusTab health={health}/>}
      </div>
    </div>);
}
