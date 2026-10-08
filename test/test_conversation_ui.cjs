// Exercise real HTML functions with a mocked transport; no robot or browser profile.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'robot_web_page.html'), 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m => m[1]).filter(s => s.trim());
for (const s of scripts) new vm.Script(s);
new vm.Script(fs.readFileSync(path.join(__dirname, '..', 'service-worker.js'), 'utf8'));
function source(name) {
  const match = new RegExp(`^  (?:async )?function ${name}\\(`, 'm').exec(html);
  assert.ok(match, name);
  const rest = html.slice(match.index);
  const end = /\n  }\r?\n/.exec(rest);
  return rest.slice(0, end.index + end[0].length);
}
function setup() {
  const saved = new Map();
  const c = vm.createContext({
    conversations: [{id:'session-A',title:'A',messages:[{id:'msg-A',taskId:'task-A'}]},
                    {id:'session-B',title:'B',messages:[{id:'msg-B',text:'B private'}]}],
    activeId:'session-B', activeTask:{id:'task-B'}, conversationStoreReady:true,
    conversationStoreLoading:false,sending:false,STORAGE_KEY:'test-history',
    pendingTurn:null,
    composerDrafts:new Map(),pendingReference:null,speechState:'idle',
    capabilitySelect:{value:''},welcomePickupAnnId:null,welcomeReturnAnnId:null,
    welcomeSelectionRole:null,mapQueryIds:new Set(),referencePreview:{hidden:true},
    patrolSelectedIds:[],patrolMapSnapshot:null,patrolRounds:{value:'-1'},refreshPatrolSetup(){},
    referencePreviewImage:{},referencePreviewName:{},clearPendingReference(){},
    refreshWelcomeSetup(){},autoResize(){},URL:{revokeObjectURL(){}},
    localStorage:{getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)},
    composerStatus:{textContent:''},uid:()=> 'session-new',input:{focus(){}},
    renderAll(){this.renders=(this.renders||0)+1},renderHistory(){},renderMapLegend(){},
    closeSidebar(){},refreshActionAvailability(){},
  });
  for (const name of ['saveConversations','loadServerConversation','initializeConversationStore',
                      'activateConversation','canSwitchConversation',
                      'newConversation','deleteConversation','refreshServerConversations','syncTask']) {
    vm.runInContext(source(name),c);
  }
  return {c,saved};
}
(async()=>{
  let tests = 0;
  {
    const {c}=setup();
    let resolve;
    c.api=()=>new Promise(r=>resolve=r);
    const pending=c.loadServerConversation('session-A');
    resolve({conversation:{id:'session-A',title:'A',messages:[{taskId:'task-A',task:{id:'task-A',status:'succeeded'}}]}});
    await pending;
    assert.equal(c.activeId,'session-B');
    assert.equal(c.activeTask.id,'task-B');
    assert.equal(c.conversations[0].messages[0].task.status,'succeeded');
    assert.equal(c.conversations[1].messages[0].text,'B private');tests++;
  }
  {
    const {c}=setup();
    c.syncTask({id:'task-A',conversation_id:'session-A',status:'succeeded'});
    assert.equal(c.activeTask.id,'task-B');
    assert.equal(c.conversations[0].messages[0].task.status,'succeeded');
    assert.equal(c.conversations[1].messages[0].task,undefined);tests++;
  }
  {
    const {c}=setup();
    c.pendingTurn={conversationId:'session-A',userId:'user-upload',assistantId:'reply-upload'};
    c.conversations[0].messages.push({id:'user-upload',text:'上传中的问题'},{id:'reply-upload',state:'thinking'});
    c.api=async()=>({conversation:{id:'session-A',messages:[]}});
    await c.loadServerConversation('session-A');
    assert.equal(c.conversations[0].messages.length,2);
    assert.equal(c.conversations[0].messages[0].text,'上传中的问题');tests++;
  }
  {
    const {c}=setup();
    c.input.value='B草稿';c.pendingReference={name:'B图片',previewUrl:'blob:b'};
    c.activateConversation('session-A');
    assert.equal(c.pendingReference,null);
    assert.equal(c.input.value,'');
    c.activateConversation('session-B');
    assert.equal(c.pendingReference.name,'B图片');
    assert.equal(c.input.value,'B草稿');tests++;
  }
  {
    const {c}=setup();c.speechState='transcribing';
    let requests=0;c.api=async()=>{requests++};
    await c.newConversation();
    assert.equal(requests,0);assert.equal(c.activeId,'session-B');tests++;
  }
  {
    const {c}=setup();
    c.api=async()=>{throw new Error('offline')};
    await c.newConversation();
    assert.equal(c.activeId,'session-B');
    assert.equal(c.conversations.length,2);tests++;
  }
  {
    const {c}=setup();
    c.api=async(p,data)=>({conversation:{id:data.id,title:data.title,messages:[]}});
    await c.newConversation();
    assert.equal(c.activeId,'session-new');
    assert.equal(c.activeTask,null);tests++;
  }
  {
    const {c,saved}=setup();
    c.conversationStoreReady=false;
    saved.set('test-history',JSON.stringify(c.conversations));
    const imports=[];
    c.api=async(p,data)=>{
      if(p.endsWith('/import')){imports.push(data.id);return {ok:true}}
      if(p==='/api/conversations')return {conversations:[{id:'session-A',title:'server A'}]};
      return {conversation:{id:'session-A',title:'server A',messages:[{text:'persistent answer'}]}};
    };
    await c.initializeConversationStore();
    assert.deepEqual(imports,['session-A','session-B']);
    assert.ok(saved.has('test-history-before-server-import'));
    assert.equal(c.activeId,'session-A');
    assert.equal(c.conversationStoreReady,true);
    await c.initializeConversationStore();
    assert.equal(imports.length,2);tests++;
  }
  {
    const {c,saved}=setup();
    c.conversationStoreReady=false;
    saved.set('test-history','original cache');
    c.api=async()=>{throw new Error('offline')};
    await c.initializeConversationStore();
    assert.equal(c.conversationStoreReady,false);
    assert.equal(c.conversations.length,2);
    assert.equal(saved.get('test-history-before-server-import'),'original cache');tests++;
  }
  {
    const {c}=setup();
    c.api=async p=>p==='/api/conversations'
      ? {conversations:[{id:'session-C',title:'from phone'}]}
      : {conversation:{id:'session-C',title:'from phone',messages:[{text:'synced'}]}};
    await c.refreshServerConversations();
    assert.equal(c.activeId,'session-C');
    assert.equal(c.conversations[0].messages[0].text,'synced');tests++;
  }
  console.log(JSON.stringify({javascript_parse:'pass',ui_contract_tests:tests,failures:0}));
})().catch(e=>{console.error(e);process.exitCode=1});
