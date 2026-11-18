import {summarizeLocation} from './locations.js';
export const CITIES=[
 {id:'bengaluru',name:'Bengaluru',subtitle:'Karnataka',areas:['JP Nagar','Whitefield','Electronic City','Yelahanka']},
 {id:'mysuru',name:'Mysuru',subtitle:'Karnataka',areas:['Vijayanagar','Hebbal','Gokulam']},
 {id:'hyderabad',name:'Hyderabad',subtitle:'Telangana',areas:['Gachibowli','Kukatpally','Madhapur']},
 {id:'chennai',name:'Chennai',subtitle:'Tamil Nadu',areas:['Velachery','Anna Nagar','Sholinganallur']},
 {id:'pune',name:'Pune',subtitle:'Maharashtra',areas:['Baner','Hinjawadi','Kharadi']},
 {id:'mumbai',name:'Mumbai',subtitle:'Maharashtra',areas:['Andheri','Powai','Borivali']}
];
export function network(cityId,area,apartment=''){
 const city=CITIES.find(c=>c.id===cityId);if(!city||!city.areas.includes(area))throw Error('Unknown city or area');
 const prefix=cityId+':'+area+':',node=(id,label,type,x,y,parent=null)=>({id:prefix+id,label,type,x,y,parent:parent?prefix+parent:null});
 if(apartment&&!['a','b','c'].includes(apartment))throw Error('Unknown apartment');
 return apartment?[
 node('transformer','Distribution transformer','Transformer',50,12),node('apartment-'+apartment,'Apartment '+apartment.toUpperCase()+' · main panel','Apartment',50,39,'transformer'),
 ...['1','2','3'].map((b,i)=>node('apartment-'+apartment+'-block-'+b,'Block '+b,'Block meter',18+i*32,75,'apartment-'+apartment))
 ]:[node('supply','Regional supply','Substation',50,10),node('feeder','Area feeder','Feeder',50,32,'supply'),node('transformer','Distribution transformer','Transformer',50,54,'feeder'),...['a','b','c'].map((a,i)=>node('apartment-'+a,'Apartment '+a.toUpperCase(),'Apartment',18+i*32,84,'transformer'))];
}
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>Number.isFinite(n)?n.toFixed(2):'—';
export function createGridExplorer(deps){
 let cityId='bengaluru',area='JP Nagar',apartment='',selected='',bindings={},message='';
 const nodes=()=>network(cityId,area,apartment),city=()=>CITIES.find(c=>c.id===cityId);
 const summary=b=>{const {map_url,...result}=summarizeLocation({rows:b.rows},deps.model());return result;};
 function page(){
 const ns=nodes();if(!ns.some(n=>n.id===selected))selected=ns.find(n=>n.type==='Apartment').id;
 const active=ns.find(n=>n.id===selected),data=bindings[selected],s=data?summary(data):null;
 const count=Object.keys(bindings).filter(k=>k.startsWith(cityId+':'+area+':')).length;
 const trail=[city().name,area,...(apartment?['Apartment '+apartment.toUpperCase()]:[])];
 const canBind=['Apartment','Block meter'].includes(active.type);
 return `<div class="page-heading"><div><span class="eyebrow">CITY → REGION → APARTMENT</span><h1>City grid explorer</h1><p>Explore the network. Select a building. Inspect its energy.</p></div><button class="btn" id="grid-export">Export network report</button></div>
 <div class="city-choices" aria-label="Choose a city">${CITIES.map(c=>`<button class="city-card ${c.id===cityId?'selected':''}" data-city="${c.id}" aria-pressed="${c.id===cityId}"><span class="city-mark">▥</span><strong>${c.name}</strong><small>${c.subtitle} · ${c.areas.length} areas</small></button>`).join('')}</div>
 <div class="grid-disclosure"><span class="tag">ILLUSTRATIVE NETWORK</span><span>Real city and locality names. Example apartments and connections—not a verified utility map. No live grid feed.</span></div>
 <div class="grid-workspace"><aside class="panel area-panel"><span class="eyebrow">${city().name}</span><h2>Explore regions</h2><div class="area-list">${city().areas.map((a,i)=>`<button data-area="${esc(a)}" class="${a===area?'selected':''}" aria-pressed="${a===area}"><span>0${i+1}</span><strong>${a}</strong><span>→</span></button>`).join('')}</div><div class="area-foot"><strong>${count}</strong><span>datasets attached in this area</span><small>Counts include main and block meters. They are not added together.</small></div><a class="text-link" href="#data">Import meter readings →</a></aside>
 <section class="panel network-panel"><div class="network-heading"><div><div class="network-trail">${trail.map(esc).join(' / ')}</div><h2>${apartment?'Apartment distribution':'Regional distribution'}</h2></div>${apartment?'<button class="btn small" id="grid-back">← All apartments</button>':'<span class="mode-chip">SCHEMATIC</span>'}</div>
 <div class="network-canvas"><svg class="network-wires" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${ns.filter(n=>n.parent).map(n=>{const p=ns.find(p=>p.id===n.parent);return `<path d="M${p.x} ${p.y} V${(p.y+n.y)/2} H${n.x} V${n.y}" class="${n.id===selected?'highlight':''}"/>`;}).join('')}</svg>${ns.map(n=>`<button class="network-node ${n.id===selected?'selected':''} ${bindings[n.id]?'attached':''}" style="left:${n.x}%;top:${n.y}%" data-node="${esc(n.id)}" aria-pressed="${n.id===selected}"><span class="node-type">${n.type==='Apartment'||n.type==='Block meter'?'▥':'ϟ'} ${n.type}</span><strong>${esc(n.label)}</strong><small>${bindings[n.id]?bindings[n.id].demo?'UCI demonstration':'CSV attached':'No telemetry'}</small></button>`).join('')}</div><div class="network-legend"><span><i></i>Illustrative connection</span><span><i class="attached"></i>Dataset attached</span><span>Select a node to inspect →</span></div></section>
 <aside class="panel node-inspector"><span class="eyebrow">SELECTED ${active.type.toUpperCase()}</span><h2>${esc(active.label)}</h2><p class="sub">${esc(area)} · ${city().name}</p><span class="tag ${data?'green':''}">${data?data.demo?'Demonstration data':'User-supplied · unverified':'Awaiting measurements'}</span>
 ${s?`<div class="node-metrics"><div><small>Latest observed load</small><strong>${fmt(s.latest_kw)} <em>kW</em></strong><span>${esc(s.last_timestamp)}</span></div><div><small>Next-hour forecast</small><strong>${fmt(s.forecast_kw)} <em>kW</em></strong><span>${esc(s.forecast_timestamp)}</span></div></div><div class="node-assessment"><strong>${s.status}</strong><p>${s.flagged_hours} flagged / ${s.scored_hours} scored hours<br>${s.observed_hours} observed / ${s.window_hours} hours</p></div><p class="small-note">${data.demo?'French household readings assigned here only to demonstrate the interface. These are not apartment or local grid measurements.':esc(data.name)+' · User-assigned meter association; location and model transfer are not independently verified.'}</p><button class="btn primary" id="grid-analyze">Open forecast & alerts →</button><button class="btn small" id="grid-unlink">Detach dataset</button>`:`<div class="no-telemetry"><span>—</span><strong>No meter readings attached</strong><p>${canBind?'Attach a CSV for this meter or explore with clearly labelled UCI demonstration readings.':'This is an example supply asset. Its load and condition cannot be inferred from a single downstream meter.'}</p></div>`}
 ${active.type==='Apartment'?`<button class="btn primary" id="grid-drill" data-apartment="${active.id.split('apartment-')[1]}">Explore apartment blocks →</button>`:''}
 ${canBind&&!data?`<button class="btn" id="grid-attach" ${deps.upload()?'':'disabled'}>Attach analyzed CSV</button><button class="btn" id="grid-demo">Try UCI demonstration</button>${!deps.upload()?'<a href="#data" class="text-link">Upload a meter CSV first →</a>':''}`:''}
 <p class="status-message" role="status">${esc(message)}</p></aside></div>
 <div class="grid-guidance"><div><strong>01 · Choose a city</strong><p>Browse major regions from the city cards.</p></div><div><strong>02 · Follow the network</strong><p>Select an apartment, then open its block meters.</p></div><div><strong>03 · Connect evidence</strong><p>Attach readings to a node and inspect dated forecasts. No data is never shown as healthy.</p></div></div>
 <p class="small-note">Connections are illustrative, not mapped electrical infrastructure or geographic positions. Attached readings stay in this tab; export before closing. The household-trained model needs local validation for apartment use. Consumption anomalies do not confirm grid faults or unsafe areas.</p>`;
 }
 function bind(){
 const on=(s,e,f)=>document.querySelector(s)?.addEventListener(e,f),redraw=()=>deps.render();
 document.querySelectorAll('[data-city]').forEach(b=>b.onclick=()=>{cityId=b.dataset.city;area=city().areas[0];apartment='';selected='';message='';redraw();});
 document.querySelectorAll('[data-area]').forEach(b=>b.onclick=()=>{area=b.dataset.area;apartment='';selected='';message='';redraw();});
 document.querySelectorAll('[data-node]').forEach(b=>b.onclick=()=>{selected=b.dataset.node;message='';redraw();});
 on('#grid-drill','click',e=>{apartment=e.currentTarget.dataset.apartment;selected=nodes().find(n=>n.type==='Block meter').id;message='';redraw();});
 on('#grid-back','click',()=>{apartment='';selected='';message='';redraw();});
 on('#grid-demo','click',()=>{bindings[selected]={rows:deps.original(),name:'UCI household demonstration',id:'original',demo:true};redraw();});
 on('#grid-attach','click',()=>{const u=deps.upload();if(!u)return;const duplicate=Object.entries(bindings).find(([id,b])=>id!==selected&&!b.demo&&b.id===u.id);if(duplicate){message='This CSV is already attached to another node. Detach it there before assigning it here.';}else{bindings[selected]={...u,demo:false};message='CSV attached to the selected meter for this session.';}redraw();});
 on('#grid-unlink','click',()=>{delete bindings[selected];message='Dataset detached.';redraw();});
 on('#grid-analyze','click',()=>deps.analyze(bindings[selected]));
 on('#grid-export','click',()=>{const output={generated_at:new Date().toISOString(),city:city().name,area,scope:'Illustrative topology, not verified utility infrastructure. No live grid feed.',nodes:network(cityId,area),apartment_blocks:['a','b','c'].map(a=>network(cityId,area,a)),readings:Object.entries(bindings).filter(([id])=>id.startsWith(cityId+':'+area+':')).map(([id,b])=>({node_id:id,file:b.name,provenance:b.demo?'UCI demonstration':'User-supplied unverified',...summary(b)}))};const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='power_guard_city_grid.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message='Network report exported.';redraw();});
 }
 return {page,bind};
}
