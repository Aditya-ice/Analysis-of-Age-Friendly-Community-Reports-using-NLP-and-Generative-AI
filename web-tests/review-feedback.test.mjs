import test from 'node:test';
import assert from 'node:assert/strict';
import {feedback,restoreDraft} from '../evaluations/v2/review_ui.mjs';
const packet={batch_id:'development-01',dataset_sha256:'dataset',index_fingerprint:'index',
  cases:[{id:'q1',sha256:'case-hash',searchable_span_ids:['span1']}]};
const entry={q1:{decision:'candidate_supported',answerability:'full',source_checked:true,selected_span_ids:['span1']}};
test('AI review is explicitly attributed and cannot export as human feedback',()=>{
  const result=feedback(packet,'Codex',true,entry,undefined,{type:'ai',model:'session',method:'direct_source_inspection'});
  assert.equal(result.format,'elderhelp-ai-ui-feedback-v1');
  assert.equal(result.review_type,'ai_reviewed');
  assert.equal(result.human_verified,undefined);
  assert.throws(()=>feedback(packet,'Codex',true,entry,undefined,{type:'ai'}));
});
test('draft resume rejects changed cases or source versions',()=>{
  const saved={binding:JSON.stringify(packet),values:[{value:'notes'}]};
  assert.deepEqual(restoreDraft(packet,saved),saved.values);
  assert.throws(()=>restoreDraft({...packet,index_fingerprint:'changed'},saved));
});
test('feedback requires human identity and source attestation',()=>{
  assert.throws(()=>feedback(packet,'',true,entry));
  assert.throws(()=>feedback(packet,'Test Reviewer',false,entry));
  assert.throws(()=>feedback(packet,'Test Reviewer',true,{q1:{...entry.q1,source_checked:false}}));
});
test('feedback remains separate from gold labels and binds exact source versions',()=>{
  const result=feedback(packet,'Test Reviewer',true,entry);
  assert.equal(result.reviews[0].case_sha256,'case-hash');
  assert.equal(result.dataset_sha256,'dataset');
  assert.equal(result.review,undefined);
  assert.equal(result.gold_evidence_units,undefined);
  assert.equal(result.reviews[0].decision,'candidate_supported');
});
test('unknown and unsearchable spans, empty support and unexplained corrections are rejected',()=>{
  for(const change of [{selected_span_ids:['invented']},{selected_span_ids:[]},{decision:'changes_needed',notes:''}]) {
    assert.throws(()=>feedback(packet,'Test Reviewer',true,{q1:{...entry.q1,...change}}));
  }
});
