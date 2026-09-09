// Exports feedback only. No network, label mutation, or automatic human-review certification.
export function feedback(packet, reviewer, attested, entries, timestamp=new Date().toISOString()) {
  reviewer=reviewer.trim();
  if(!reviewer || !attested) throw Error('Enter your name and confirm you personally checked the original sources.');
  const reviews=[];
  for(const c of packet.cases) {
    const entry=entries[c.id];
    if(!entry || entry.decision==='pending') continue;
    const {decision,answerability,notes='',selected_span_ids:spans=[],source_checked}=entry;
    if(!['candidate_supported','changes_needed'].includes(decision)) throw Error(`${c.id}: unknown decision.`);
    if(!source_checked || !['full','partial','insufficient','clarification_required'].includes(answerability)) {
      throw Error(`${c.id}: check the original source and choose answerability.`);
    }
    if(decision==='changes_needed' && !notes.trim()) throw Error(`${c.id}: explain the correction needed.`);
    if(spans.some(s=>!c.searchable_span_ids.includes(s))) throw Error(`${c.id}: unknown or unsearchable passage.`);
    if(decision==='candidate_supported' && ['full','partial'].includes(answerability) && !spans.length) {
      throw Error(`${c.id}: select the supporting searchable passages.`);
    }
    reviews.push({case_id:c.id,case_sha256:c.sha256,decision,answerability,selected_span_ids:[...new Set(spans)],notes:notes.trim()});
  }
  if(!reviews.length) throw Error('Review at least one case. Unfinished cases can stay pending.');
  return {format:'elderhelp-human-feedback-v1',batch_id:packet.batch_id,dataset_sha256:packet.dataset_sha256,
    index_fingerprint:packet.index_fingerprint,reviewer,reviewed_at:timestamp,original_sources_checked:true,reviews};
}

if(typeof document!=='undefined') {
  const packet=JSON.parse(document.querySelector('#packet').textContent);
  document.querySelector('#export').addEventListener('click',()=>{
    const status=document.querySelector('#status');
    try {
      const entries={};
      for(const c of packet.cases) {
        const box=document.getElementById(c.id);
        entries[c.id]={decision:box.querySelector('.decision').value,
          answerability:box.querySelector('.answerability').value,
          notes:box.querySelector('textarea').value,
          selected_span_ids:[...box.querySelectorAll('.span:checked')].map(s=>s.value),
          source_checked:box.querySelector('.source-checked').checked};
      }
      const result=feedback(packet,document.querySelector('#reviewer').value,document.querySelector('#attest').checked,entries);
      const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
      const link=document.createElement('a');link.href=url;link.download=packet.batch_id+'-feedback.json';link.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
      status.textContent='Feedback exported. Gold labels remain unchanged. Keep this file privately; unfinished form entries are not saved after reload.';
    } catch(error) {status.textContent=error.message;}
  });
}
