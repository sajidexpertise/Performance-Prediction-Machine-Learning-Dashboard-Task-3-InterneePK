const fs=require('fs'),vm=require('vm'),assert=require('assert');
const path=require('path');const root=path.join(__dirname,'..');
const b=vm.runInNewContext(fs.readFileSync(path.join(root,'dist/bundle.js'),'utf8')+';BUNDLE');
const m=b.model;
for(const r of b.records){const z=m.intercept+m.features.reduce((s,k,i)=>s+(r[k]-m.mean[i])/m.scale[i]*m.coef[i],0);const p=1/(1+Math.exp(-z));assert(Math.abs(p-r.success_probability)<1e-12);assert(p>=0&&p<=1)}
assert(b.records.filter(r=>r.split==='test').length===b.metrics.test_count);
console.log('PASS: JavaScript parity on all records; bounded probabilities; holdout count');
