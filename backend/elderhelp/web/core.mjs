export const statusLabel = status => ({grounded:'Supported by report evidence',partial:'Partial answer · some evidence is missing',insufficient_evidence:'Insufficient evidence',clarification_required:'Clarification needed'}[status] || 'Unknown answer status');
export function messageFor(status, retry) {
  if(status===401) return 'Pilot access expired or is invalid. Enter your invite code again. Saved history is still available.';
  if(status===429) return `The pilot has reached a usage limit. Try keyword search, or retry after ${retry || '60'} seconds.`;
  if(status===413 || status===422) return 'Check your question: use 1–2,000 characters and valid filters.';
  if(status===503 || status===504) return 'The service is unavailable or waking from sleep. Wait a moment and retry. Saved history remains available.';
  return 'The connection failed. Check your network and retry; your saved history remains available offline.';
}
export function requestFor(question, turns, followup, report) {
  return {question:question.trim(),history:followup?turns.slice(-6):[],filters:{report_ids:report?[report]:[]}};
}
export class SSEParser {
  buffer=''; name='message'; data=[]; dataLength=0;
  feed(text) {
    this.buffer+=text;
    if(this.buffer.length>131072) throw new Error('Answer stream exceeded the message limit.');
    const output=[]; let end;
    while((end=this.buffer.indexOf('\n'))>=0) {
      const line=this.buffer.slice(0,end).replace(/\r$/,''); this.buffer=this.buffer.slice(end+1);
      if(!line) { if(this.data.length) output.push({name:this.name,data:JSON.parse(this.data.join('\n'))});this.name='message';this.data=[];this.dataLength=0;continue; }
      if(line[0]===':') continue;
      const split=line.indexOf(':'); const field=split<0?line:line.slice(0,split); const value=split<0?'':line.slice(split+1).replace(/^ /,'');
      if(field==='event') this.name=value;
      if(field==='data') {
        this.dataLength+=value.length+1;
        if(this.dataLength>131072) throw new Error('Answer stream exceeded the message limit.');
        this.data.push(value);
      }
    }
    return output;
  }
}
export class AnswerGate {
  started=false; done=false; delta='';
  accept(event) {
    if(this.done) throw new Error('Unexpected data after completion.');
    if(event.name==='start') {if(this.started) throw new Error('Duplicate start.');this.started=true;return null;}
    if(!this.started) throw new Error('Missing answer start.');
    if(event.name==='progress') return null;
    if(event.name==='error') throw new Error(event.data.code==='quota_exhausted'?messageFor(429):messageFor(503));
    if(event.name==='delta') {this.delta+=event.data.text;if(this.delta.length>64000) throw new Error('Answer is too long.');return null;}
    if(event.name==='complete') {
      const result=event.data;
      if(!['grounded','partial','insufficient_evidence','clarification_required'].includes(result.status) || !Array.isArray(result.citations) || typeof result.answer_markdown!=='string') throw new Error('Unreadable completion.');
      if(this.delta && this.delta!==result.answer_markdown) throw new Error('The completed answer differs from its stream. Please retry.');
      this.done=true;return result;
    }
    throw new Error('Unknown answer event.');
  }
}
export const safeURL = value => {try {const url=new URL(value);return url.protocol==='https:'?url.href:null;}catch{return null;}};
