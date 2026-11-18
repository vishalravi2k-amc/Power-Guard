// Portable inference for the project's trained PyTorch and sklearn artifacts.
// No uploaded model code or pickles are executed in the browser.
export const HOUR=3600000;
export const finite=v=>typeof v==='number' && Number.isFinite(v);
export function dateOf(t){return new Date(t.replace(' ','T').replace(/Z$/,'')+'Z');}
export function stamp(d){return d.toISOString().slice(0,19).replace('T',' ');}
export function calendar(t){const d=dateOf(t),h=d.getUTCHours(),day=(d.getUTCDay()+6)%7;return [Math.sin(2*Math.PI*h/24),Math.cos(2*Math.PI*h/24),Math.sin(2*Math.PI*day/7),Math.cos(2*Math.PI*day/7)];}
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const linear=(x,w,b,relu=false)=>w.map((r,i)=>{const v=dot(r,x)+b[i];return relu?Math.max(0,v):v;});
const sigmoid=x=>1/(1+Math.exp(-x));
export function forecast(history,target,model){
  const m=model.meta,w=model.lstm;
  if(history.length<48||history.slice(-48).some(r=>!finite(r.load_kw)))return null;
  let h=new Array(32).fill(0),c=new Array(32).fill(0);
  for(const r of history.slice(-48)){
    const x=[(r.load_kw-m.load_mean)/m.load_std,...calendar(r.timestamp)];
    const gates=w['lstm.weight_ih_l0'].map((row,i)=>dot(row,x)+dot(w['lstm.weight_hh_l0'][i],h)+w['lstm.bias_ih_l0'][i]+w['lstm.bias_hh_l0'][i]);
    c=c.map((v,j)=>sigmoid(gates[32+j])*v+sigmoid(gates[j])*Math.tanh(gates[64+j]));
    h=c.map((v,j)=>sigmoid(gates[96+j])*Math.tanh(v));
  }
  const a=linear([...h,...calendar(target)],w['head.0.weight'],w['head.0.bias'],true);
  return Math.max(0,linear(a,w['head.2.weight'],w['head.2.bias'])[0]*m.load_std+m.load_mean);
}
function pathLength(n){return n<=1?0:n===2?1:2*(Math.log(n-1)+0.5772156649015329)-2*(n-1)/n;}
export function scores(rows,i,model){
  if(i<24||!rows[i].observed)return {ae_score:null,if_score:null};
  const p=rows.slice(i-24,i).map(r=>r.load_kw),v=rows[i].load_kw;
  if(!finite(v)||p.some(x=>!finite(x)))return {ae_score:null,if_score:null};
  const mean=p.reduce((a,b)=>a+b,0)/24,std=Math.sqrt(p.reduce((a,b)=>a+(b-mean)**2,0)/23),cal=calendar(rows[i].timestamp);
  const day=(dateOf(rows[i].timestamp).getUTCDay()+6)%7;
  const f=[v,v-p[23],v-p[0],mean,std,cal[0],cal[1],day>=5?1:0];
  const z=f.map((x,j)=>Math.fround((x-model.meta.feature_mean[j])/model.meta.feature_std[j]));
  let a=z;
  for(const n of [0,2,4,6])a=linear(a,model.autoencoder[`net.${n}.weight`],model.autoencoder[`net.${n}.bias`],n!==6);
  const ae=a.reduce((s,v,j)=>s+(v-z[j])**2,0)/8;
  let depths=0;
  for(const tree of model.forest.trees){let node=0,depth=0;while(tree.left[node]!==-1){node=z[tree.features[tree.feature[node]]]<=tree.threshold[node]?tree.left[node]:tree.right[node];depth++;}depths+=depth+pathLength(tree.count[node]);}
  const score=2**(-depths/(model.forest.trees.length*pathLength(model.forest.max_samples)));
  return {ae_score:ae,if_score:score};
}
export function predictRow(rows,i,model){const r=rows[i],s=scores(rows,i,model);return {...r,...s,forecast_kw:r.observed&&finite(r.load_kw)?forecast(rows.slice(Math.max(0,i-48),i),r.timestamp,model):null,persistence_kw:i?rows[i-1].load_kw:null,previous_day_kw:i>=24?rows[i-24].load_kw:null,ae_alert:finite(s.ae_score)&&s.ae_score>model.meta.ae_threshold,if_alert:finite(s.if_score)&&s.if_score>model.meta.if_threshold};}
export function parseCSV(text){
  if(new TextEncoder().encode(text).length>10*1024*1024)throw Error('Use a CSV smaller than 10 MB.');
  const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/),header=lines.shift().split(',').map(s=>s.trim().replace(/^"|"$/g,''));
  const ti=header.indexOf('timestamp'),li=header.indexOf('load_kw');
  if(ti<0||li<0)throw Error('CSV needs timestamp and load_kw columns.');
  if(lines.length<72||lines.length>50000)throw Error('Provide 72 to 50,000 hourly rows.');
  const seen=new Set();
  const rows=lines.map((line,i)=>{
    const fields=line.split(',').map(s=>s.trim().replace(/^"|"$/g,'')),t=fields[ti],value=fields[li];
    if(!/^\d{4}-\d{2}-\d{2}[ T]\d{2}:00(?::00)?$/.test(t||''))throw Error(`Row ${i+2}: use exact-hour local timestamps, e.g. 2026-11-01 09:00:00.`);
    const normalized=t.replace('T',' ')+(t.length===16?':00':''),d=dateOf(normalized),ms=d.getTime();
    if(!Number.isFinite(ms)||stamp(d)!==normalized)throw Error(`Row ${i+2}: invalid calendar date.`);
    if(seen.has(ms))throw Error(`Row ${i+2}: duplicate timestamp.`);seen.add(ms);
    if(!value||!Number.isFinite(Number(value))||Number(value)<0)throw Error(`Row ${i+2}: load_kw must be a finite non-negative number.`);
    return {timestamp:normalized,load_kw:Number(value),observed:true,injected:false,event_type:'Uploaded reading'};
  }).sort((a,b)=>a.timestamp.localeCompare(b.timestamp));
  if(dateOf(rows.at(-1).timestamp)-dateOf(rows[0].timestamp)>5*366*24*HOUR)throw Error('Limit the file to a five-year period.');
  const result=[];let previous=null,run=0,best=0;
  for(const r of rows){if(previous){let gap=0;for(let t=dateOf(previous.timestamp).getTime()+HOUR;t<dateOf(r.timestamp).getTime();t+=HOUR){gap++;result.push({timestamp:stamp(new Date(t)),load_kw:gap<=3?previous.load_kw:null,observed:false,injected:false,event_type:'Missing reading'});}}
    result.push(r);previous=r;
  }
  for(const r of result){run=finite(r.load_kw)?run+1:0;best=Math.max(best,run);}
  if(best<49)throw Error('At least 49 consecutive usable hours are needed.');
  return result;
}
