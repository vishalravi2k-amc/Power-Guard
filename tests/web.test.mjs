import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {forecast,scores,predictRow,parseCSV,HOUR,stamp,dateOf} from '../web/engine.js';
const model=JSON.parse(readFileSync('web/models/weights.json'));
const golden=JSON.parse(readFileSync('tests/golden_cases.json'));
const close=(a,b,tol=0.00002)=>assert.ok(Math.abs(a-b)<tol,`${a} differs from ${b}`);
for(const [i,c] of golden.entries())test(`Portable inference agrees with Python case ${i+1}`,()=>{
  const rows=c.rows.map(r=>({...r,timestamp:r.timestamp.replace('T',' ').slice(0,19)}));
  const result=predictRow(rows,48,model);
  for(const key of ['forecast_kw','ae_score','if_score'])close(result[key],c.expected[key]);
  close(forecast(rows,stamp(new Date(dateOf(rows.at(-1).timestamp).getTime()+HOUR)),model),c.next);
});
const sample=readFileSync('data/sample_upload.csv','utf8');
test('Valid sample parses and runs actual models',()=>{const r=parseCSV(sample);assert.ok(r.length>=72);const p=predictRow(r,50,model);assert.ok(Number.isFinite(p.forecast_kw));assert.ok(Number.isFinite(p.ae_score));});
test('Forecast excludes the target and future observations',()=>{const r=parseCSV(sample),base=predictRow(r,50,model).forecast_kw;r[50].load_kw=999;r[51].load_kw=999;close(predictRow(r,50,model).forecast_kw,base,1e-10);});
test('Reject malformed, duplicate and nonnumeric records',()=>{assert.throws(()=>parseCSV('bad,columns\n1,2'),/timestamp/);assert.throws(()=>parseCSV(sample.replace(/,\d[^\n]*\n/,',-1\n')),/non-negative/);const lines=sample.trim().split('\n');lines[2]=lines[1];assert.throws(()=>parseCSV(lines.join('\n')),/duplicate/);});
test('Reject impossible dates and timezone offsets',()=>{let lines=sample.trim().split('\n');lines[1]='2010-02-30 00:00:00,1';assert.throws(()=>parseCSV(lines.join('\n')),/invalid calendar date/);lines[1]='2010-01-01T00:00:00Z,1';assert.throws(()=>parseCSV(lines.join('\n')),/exact-hour/);});
test('Missing hours are causal and stop filling at three',()=>{const lines=sample.trim().split('\n');lines.splice(10,5);const rows=parseCSV(lines.join('\n'));assert.equal(rows[9].observed,false);assert.equal(rows[9].load_kw,rows[8].load_kw);assert.equal(rows[12].load_kw,null);assert.equal(forecast(rows.slice(0,49),rows[49].timestamp,model),null);});
test('Historical metrics recompute from the exact eligible timestamps',()=>{
 const rows=JSON.parse(readFileSync('web/data/original.json')).filter(r=>r.timestamp>=model.meta.test_first&&Number.isFinite(r.forecast_kw)&&Number.isFinite(r.persistence_kw)&&Number.isFinite(r.previous_day_kw));
 const metrics=JSON.parse(readFileSync('reports/metrics.json'));
 ['forecast_kw','persistence_kw','previous_day_kw'].forEach((key,i)=>{assert.equal(rows.length,metrics.forecast[i].samples);close(rows.reduce((s,r)=>s+Math.abs(r.load_kw-r[key]),0)/rows.length,metrics.forecast[i].mae_kw,1e-9);});
});
test('Both challenge confusion matrices match saved measured metrics',()=>{
 const rows=JSON.parse(readFileSync('web/data/challenge.json')).filter(r=>r.timestamp>=model.meta.test_first&&Number.isFinite(r.ae_score)&&Number.isFinite(r.if_score));
 const m=JSON.parse(readFileSync('reports/metrics.json'));
 for(const [prefix,name] of [['ae','Autoencoder'],['if','Isolation Forest']]){const counts={tp:0,fp:0,fn:0,tn:0};for(const r of rows){const positive=r[`${prefix}_score`]>model.meta[`${prefix}_threshold`];counts[r.injected?(positive?'tp':'fn'):(positive?'fp':'tn')]++;}for(const k of Object.keys(counts))assert.equal(counts[k],m.injected_event_detection[name][k]);}
});

const {validateLocation,summarizeLocation}=await import('../web/locations.js');
test('Location metadata requires explicit valid coordinates and provenance',()=>{
 assert.throws(()=>validateLocation({name:'A',area:'B',latitude:'',longitude:'77',provenance:'local'}),/both GPS/);
 assert.throws(()=>validateLocation({name:'A',area:'B',latitude:91,longitude:77,provenance:'local'}),/valid latitude/);
 const m=validateLocation({name:' A ',area:' JP Nagar ',latitude:'12.9',longitude:'77.6',provenance:'demonstration'});assert.equal(m.name,'A');assert.equal(m.provenance,'demonstration');
});
test('Location forecast is relative to data end and scores preserve missing evidence',()=>{
 const rows=JSON.parse(readFileSync('web/data/original.json')).slice(48,120);
 const site={rows,latitude:12.9,longitude:77.6};const s=summarizeLocation(site,model);
 assert.equal(dateOf(s.forecast_timestamp)-dateOf(rows.at(-1).timestamp),HOUR);
 assert.equal(s.window_hours,24);assert.ok(s.map_url.includes('mlat=12.9'));
 rows.at(-1).ae_score=null;assert.equal(summarizeLocation(site,model).status,'Incomplete evidence');
});

test('City grids isolate meter identities and preserve parents through apartment drilldown',async()=>{
 const {CITIES,network}=await import('../web/grid.js');
 const ids=new Set();
 for(const city of CITIES)for(const area of city.areas){
  const region=network(city.id,area);
  for(const n of region){assert.ok(!ids.has(n.id));ids.add(n.id);if(n.parent)assert.ok(region.some(p=>p.id===n.parent));}
  for(const a of ['a','b','c']){
   const detail=network(city.id,area,a),main=detail.find(n=>n.type==='Apartment');
   assert.ok(region.some(n=>n.id===main.id));
   for(const n of detail.filter(n=>n.type==='Block meter')){assert.equal(n.parent,main.id);assert.ok(!ids.has(n.id));ids.add(n.id);}
  }
 }
 assert.throws(()=>network('missing','JP Nagar'));assert.throws(()=>network('bengaluru','missing'));
});
