import {parseCSV,predictRow} from './engine.js';
self.onmessage=async({data})=>{
  try{
    const rows=parseCSV(data.text),model=await fetch('./models/weights.json').then(r=>{if(!r.ok)throw Error('Model could not be loaded.');return r.json();});
    const results=[];
    for(let i=0;i<rows.length;i++){results.push(predictRow(rows,i,model));if(i%100===0)self.postMessage({progress:Math.round(i/rows.length*100)});}
    self.postMessage({rows:results});
  }catch(e){self.postMessage({error:e.message});}
};
