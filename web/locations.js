import {finite,forecast,dateOf,stamp,HOUR} from './engine.js';
export function validateLocation({name,area,latitude,longitude,provenance}){
 name=String(name||'').trim();area=String(area||'').trim();
 if(!name||!area)throw Error('Enter a meter/site name and locality.');
 if(name.length>100||area.length>100)throw Error('Keep names under 100 characters.');
 if(String(latitude).trim()===''||String(longitude).trim()==='')throw Error('Enter both GPS coordinates. Area names alone cannot identify a meter.');
 const lat=Number(latitude),lon=Number(longitude);
 if(!Number.isFinite(lat)||!Number.isFinite(lon)||lat < -90||lat>90||lon < -180||lon>180)throw Error('Use valid latitude (−90 to 90) and longitude (−180 to 180).');
 if(!['local','demonstration'].includes(provenance))throw Error('Choose the origin of the readings.');
 return {name,area,latitude:lat,longitude:lon,provenance};
}
export function summarizeLocation(site,model){
 const last=site.rows.at(-1),window=site.rows.slice(-24),eligible=window.filter(r=>finite(r.ae_score)&&finite(r.if_score));
 const flagged=eligible.filter(r=>r.ae_score>model.meta.ae_threshold||r.if_score>model.meta.if_threshold);
 const target=stamp(new Date(dateOf(last.timestamp).getTime()+HOUR));
 return {last_timestamp:last.timestamp,forecast_timestamp:target,forecast_kw:forecast(site.rows.slice(-48),target,model),latest_kw:last.load_kw,
 observed_hours:window.filter(r=>r.observed&&finite(r.load_kw)).length,window_hours:window.length,scored_hours:eligible.length,flagged_hours:flagged.length,
 status:eligible.length<24?'Incomplete evidence':flagged.length?'Review suggested':'No flags in window',
 map_url:`https://www.openstreetmap.org/?mlat=${site.latitude}&mlon=${site.longitude}#map=16/${site.latitude}/${site.longitude}`};
}
