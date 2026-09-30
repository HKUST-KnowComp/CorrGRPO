'use strict';
const byId = id => document.getElementById(id);
let domain='code';const tabs=[...document.querySelectorAll('[data-domain]')];
function renderResults(){const d=experimentData[domain];const modelIndex=Number(byId('model').value);const rows=d.rows[modelIndex];
 const best=d.headers.map((_,j)=>(d.lower.includes(j)?Math.min:Math.max)(...rows.map(r=>r[j])));
 const corrIndex=d.methods.indexOf('CorrGRPO');
 const winners=best.map((v,j)=>corrIndex>=0&&rows[corrIndex][j]===v?corrIndex:rows.filter(r=>r[j]===v).length===1?rows.findIndex(r=>r[j]===v):-1);
 byId('result-caption').innerHTML='<span class="caption-title">'+d.description+' · '+d.models[modelIndex]+'</span><span class="caption-metrics">'+d.note+'</span>';
 byId('result-table').innerHTML='<caption class="sr-only">'+d.description+' · '+d.models[modelIndex]+'</caption><thead><tr><th scope="col">Method</th>'+d.headers.map(h=>'<th scope="col">'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map((r,i)=>'<tr class="'+(d.methods[i]==='CorrGRPO'?'corr':'')+'"><th scope="row">'+d.methods[i]+'</th>'+r.map((v,j)=>'<td class="'+(i===winners[j]?'best':'')+'">'+v.toFixed(2)+'</td>').join('')+'</tr>').join('')+'</tbody>';
}
function setDomain(name){domain=name;tabs.forEach(t=>{const active=t.dataset.domain===name;t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1});byId('result-panel').setAttribute('aria-labelledby','tab-'+name);byId('model').innerHTML=experimentData[name].models.map((m,i)=>'<option value="'+i+'">'+m+'</option>').join('');if(name==='code')byId('model').value=3;renderResults();}
tabs.forEach((t,i)=>{t.addEventListener('click',()=>setDomain(t.dataset.domain));t.addEventListener('keydown',e=>{let j;if(e.key==='ArrowRight')j=(i+1)%tabs.length;if(e.key==='ArrowLeft')j=(i+tabs.length-1)%tabs.length;if(e.key==='Home')j=0;if(e.key==='End')j=tabs.length-1;if(j!==undefined){e.preventDefault();tabs[j].focus();setDomain(tabs[j].dataset.domain)}})});
byId('model').addEventListener('change',renderResults);setDomain('code');
byId('copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(byId('bibtex').textContent);byId('copy-status').textContent='BibTeX copied.';byId('copy-citation').textContent='Copied';setTimeout(()=>byId('copy-citation').textContent='Copy',2000)}catch{const range=document.createRange();range.selectNodeContents(byId('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);byId('copy-status').textContent='Citation selected. Press Ctrl+C or ⌘C to copy.'}});

const scale=byId('scale');
const exampleRewards=[[.1,.18,.8],[.18,.1,7.28],[.82,.9,8],[.9,.82,1.52]];
function updateMatrices(){
 const factor=Number(scale.value);byId('scale-value').textContent=factor.toFixed(2)+'×';scale.setAttribute('aria-valuetext',factor.toFixed(2)+' times');
 const rewards=exampleRewards.map(row=>[row[0],row[1],row[2]*factor]);
 const mean=[0,1,2].map(j=>rewards.reduce((sum,row)=>sum+row[j],0)/rewards.length);
 const cov=[0,1,2].map(i=>[0,1,2].map(j=>rewards.reduce((sum,row)=>sum+(row[i]-mean[i])*(row[j]-mean[j]),0)/(rewards.length-1)));
 const corr=cov.map((row,i)=>row.map((v,j)=>v/Math.sqrt(cov[i][i]*cov[j][j])));
 for(const [id,matrix,normalizer,label] of [['cov-matrix',cov,14.1696,'Covariance'],['corr-matrix',corr,1,'Correlation']]){
  const header='<span class="matrix-axis"></span>'+[1,2,3].map(n=>'<span class="matrix-axis">r'+n+'</span>').join('');
  byId(id).innerHTML=header+matrix.map((row,i)=>'<span class="matrix-axis">r'+(i+1)+'</span>'+row.map(v=>'<span class="matrix-value" style="background:rgba(39,99,142,'+(.045+.24*Math.min(v/normalizer,1))+')">'+v.toFixed(3)+'</span>').join('')).join('');
  byId(id).setAttribute('aria-label',label+' matrix. '+matrix.map((row,i)=>'Row '+(i+1)+': '+row.map(v=>v.toFixed(3)).join(', ')).join('. '));
 }
 byId('grpo-mult').textContent=(1/Math.sqrt(cov.flat().reduce((a,b)=>a+b,0))).toFixed(4)+'×';
 byId('corr-mult').textContent=(1/Math.sqrt(corr.flat().reduce((a,b)=>a+b,0))).toFixed(4)+'×';
}
scale.addEventListener('input',updateMatrices);byId('reset').addEventListener('click',()=>{scale.value='1';updateMatrices()});updateMatrices();
